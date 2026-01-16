"""
VAMOS Paper Benchmark Script
===========================
Runs complete benchmark and writes CSV results for the paper.

Usage: python paper/01_run_paper_benchmark.py

Use 04_update_paper_tables_from_csv.py to generate and inject LaTeX tables.

Environment variables:
  - VAMOS_N_EVALS: evaluations per run (default: 100000)
  - VAMOS_N_SEEDS: number of seeds (default: 30)
  - VAMOS_N_JOBS: joblib workers (default: CPU count - 1)
  - VAMOS_PAPER_FRAMEWORKS: comma-separated framework keys (default: all)
  - VAMOS_PAPER_OUTPUT_CSV: output CSV path (default: experiments/benchmark_paper.csv)
  - VAMOS_NUMBA_WARMUP_EVALS: warmup evaluations before timing Numba runs (default: 2000)
  - VAMOS_OBJECTIVE_ALIGNMENT_CHECK: 1/0 run objective-alignment preflight (default: 1)
  - VAMOS_OBJECTIVE_ALIGNMENT_SAMPLES: random points per problem (default: 64)
  - VAMOS_OBJECTIVE_ALIGNMENT_RTOL: relative tolerance (default: 1e-6)
  - VAMOS_OBJECTIVE_ALIGNMENT_ATOL: absolute tolerance (default: 1e-8)
"""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

# Add project src to path
sys.path.insert(0, str(ROOT_DIR / "src"))

# Prefer local checkouts when available
DESKTOP_DIR = ROOT_DIR.parent
JMETALPY_SRC = DESKTOP_DIR / "jMetalPy" / "src"
PLATYPUS_SRC = DESKTOP_DIR / "Platypus"
for extra_path in (JMETALPY_SRC, PLATYPUS_SRC):
    if extra_path.exists():
        sys.path.insert(0, str(extra_path))

import time
import pandas as pd
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "experiments"

# Problems to benchmark (by family)
ZDT_PROBLEMS = ["zdt1", "zdt2", "zdt3", "zdt4", "zdt6"]
DTLZ_PROBLEMS = ["dtlz1", "dtlz2", "dtlz3", "dtlz4", "dtlz5", "dtlz6", "dtlz7"]
WFG_PROBLEMS = ["wfg1", "wfg2", "wfg3", "wfg4", "wfg5", "wfg6", "wfg7", "wfg8", "wfg9"]

USE_ZDT = True
USE_DTLZ = True
USE_WFG = True

N_EVALS = int(os.environ.get("VAMOS_N_EVALS", "100000"))
N_SEEDS = int(os.environ.get("VAMOS_N_SEEDS", "30"))
OUTPUT_CSV = Path(os.environ.get("VAMOS_PAPER_OUTPUT_CSV", str(DATA_DIR / "benchmark_paper.csv")))
NUMBA_WARMUP_EVALS = int(os.environ.get("VAMOS_NUMBA_WARMUP_EVALS", "2000"))
ALIGN_CHECK_ENABLED = bool(int(os.environ.get("VAMOS_OBJECTIVE_ALIGNMENT_CHECK", "1")))
ALIGN_CHECK_SAMPLES = int(os.environ.get("VAMOS_OBJECTIVE_ALIGNMENT_SAMPLES", "64"))
ALIGN_CHECK_RTOL = float(os.environ.get("VAMOS_OBJECTIVE_ALIGNMENT_RTOL", "1e-6"))
ALIGN_CHECK_ATOL = float(os.environ.get("VAMOS_OBJECTIVE_ALIGNMENT_ATOL", "1e-8"))

POP_SIZE = 100
CROSSOVER_PROB = 1.0
CROSSOVER_ETA = 20.0
MUTATION_ETA = 20.0

ZDT_N_VAR = {"zdt1": 30, "zdt2": 30, "zdt3": 30, "zdt4": 10, "zdt6": 10}
DTLZ_N_VAR = {"dtlz1": 7, "dtlz2": 12, "dtlz3": 12, "dtlz4": 12, "dtlz5": 12, "dtlz6": 12, "dtlz7": 22}
WFG_N_VAR = 24
ZDT_N_OBJ = 2
DTLZ_N_OBJ = 3
WFG_N_OBJ = 2

# Frameworks to benchmark
DEFAULT_FRAMEWORKS = [
    "vamos-numpy",
    "vamos-numba",
    "vamos-moocore",  # VAMOS backends
    "pymoo",  # pymoo
    "deap",  # DEAP
    "jmetalpy",  # jMetalPy
    "platypus",  # Platypus
]
_frameworks_env = os.environ.get("VAMOS_PAPER_FRAMEWORKS")
FRAMEWORKS = [f.strip() for f in _frameworks_env.split(",") if f.strip()] if _frameworks_env else DEFAULT_FRAMEWORKS

# Build problem list
PROBLEMS = []
if USE_ZDT:
    PROBLEMS.extend(ZDT_PROBLEMS)
if USE_DTLZ:
    PROBLEMS.extend(DTLZ_PROBLEMS)
if USE_WFG:
    PROBLEMS.extend(WFG_PROBLEMS)

print(f"Configured {len(PROBLEMS)} problems: {PROBLEMS}")
print(f"Frameworks: {FRAMEWORKS}")
print(f"Evaluations per run: {N_EVALS:,}")
print(f"Seeds: {N_SEEDS}")
print(f"Total runs: {len(PROBLEMS) * len(FRAMEWORKS) * N_SEEDS}")

# =============================================================================
# PARALLEL BENCHMARK EXECUTION
# =============================================================================

from joblib import Parallel, delayed

# Use all cores minus 1
N_JOBS = int(os.environ.get("VAMOS_N_JOBS", max(1, os.cpu_count() - 1)))
print(f"Using {N_JOBS} parallel workers")

from vamos.foundation.problem.registry import make_problem_selection
from vamos import optimize
from vamos.engine.algorithm.config import NSGAIIConfig

try:
    from .benchmark_utils import compute_hv
except ImportError:
    from benchmark_utils import compute_hv

try:
    from .progress_utils import ProgressBar, joblib_progress
except ImportError:  # pragma: no cover
    from progress_utils import ProgressBar, joblib_progress


# =============================================================================
# PROBLEM DIMENSIONS


def problem_dims(problem_name: str) -> tuple[int, int]:
    if problem_name in ZDT_N_VAR:
        return ZDT_N_VAR[problem_name], ZDT_N_OBJ
    if problem_name in DTLZ_N_VAR:
        return DTLZ_N_VAR[problem_name], DTLZ_N_OBJ
    if problem_name in WFG_PROBLEMS:
        return WFG_N_VAR, WFG_N_OBJ
    raise ValueError(f"Unknown problem dimensions for '{problem_name}'")


# =============================================================================
# OBJECTIVE ALIGNMENT CHECKS (cross-framework validity)


def _as_1d_bounds(xl: object, xu: object, n_var: int) -> tuple[np.ndarray, np.ndarray]:
    xl_arr = np.asarray(xl, dtype=float)
    xu_arr = np.asarray(xu, dtype=float)
    if xl_arr.ndim == 0:
        xl_arr = np.full(n_var, float(xl_arr))
    if xu_arr.ndim == 0:
        xu_arr = np.full(n_var, float(xu_arr))
    if xl_arr.shape != (n_var,) or xu_arr.shape != (n_var,):
        raise ValueError(f"Invalid bounds: xl={xl_arr.shape}, xu={xu_arr.shape}")
    return xl_arr, xu_arr


def run_objective_alignment_checks() -> None:
    if not ALIGN_CHECK_ENABLED:
        print("Objective alignment check disabled (VAMOS_OBJECTIVE_ALIGNMENT_CHECK=0)")
        return

    selected = set(FRAMEWORKS)
    check_pymoo = "pymoo" in selected
    check_jmetal = "jmetalpy" in selected
    check_platypus = "platypus" in selected

    if not (check_pymoo or check_jmetal or check_platypus):
        print("Objective alignment check skipped (no external frameworks selected)")
        return

    n_samples = max(1, int(ALIGN_CHECK_SAMPLES))
    print("\nObjective alignment preflight")
    print(f"- samples per problem: {n_samples}")
    print(f"- tolerance: rtol={ALIGN_CHECK_RTOL:g}, atol={ALIGN_CHECK_ATOL:g}")

    rng = np.random.default_rng(12345)

    # Optional imports (only when selected)
    pymoo_get_problem = None
    if check_pymoo:
        try:
            from pymoo.problems import get_problem as pymoo_get_problem  # type: ignore[assignment]
        except Exception as e:  # pragma: no cover
            print(f"  Warning: pymoo alignment check skipped (import failed): {e}")
            check_pymoo = False

    jmetal_problem_map = None
    if check_jmetal:
        try:
            from jmetal.problem import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
            from jmetal.problem import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7
            from jmetal.problem import WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9

            # jMetalPy's WFG defaults use k=2 for m=2, while VAMOS/pymoo use k=4.
            # To avoid cross-toolkit semantic drift, pass (k,l) explicitly.
            wfg_k = 4 if WFG_N_OBJ == 2 else 2 * (WFG_N_OBJ - 1)
            wfg_l = WFG_N_VAR - wfg_k

            jmetal_problem_map = {
                "zdt1": ZDT1(number_of_variables=ZDT_N_VAR["zdt1"]),
                "zdt2": ZDT2(number_of_variables=ZDT_N_VAR["zdt2"]),
                "zdt3": ZDT3(number_of_variables=ZDT_N_VAR["zdt3"]),
                "zdt4": ZDT4(number_of_variables=ZDT_N_VAR["zdt4"]),
                "zdt6": ZDT6(number_of_variables=ZDT_N_VAR["zdt6"]),
                "dtlz1": DTLZ1(number_of_variables=DTLZ_N_VAR["dtlz1"], number_of_objectives=DTLZ_N_OBJ),
                "dtlz2": DTLZ2(number_of_variables=DTLZ_N_VAR["dtlz2"], number_of_objectives=DTLZ_N_OBJ),
                "dtlz3": DTLZ3(number_of_variables=DTLZ_N_VAR["dtlz3"], number_of_objectives=DTLZ_N_OBJ),
                "dtlz4": DTLZ4(number_of_variables=DTLZ_N_VAR["dtlz4"], number_of_objectives=DTLZ_N_OBJ),
                "dtlz5": DTLZ5(number_of_variables=DTLZ_N_VAR["dtlz5"], number_of_objectives=DTLZ_N_OBJ),
                "dtlz6": DTLZ6(number_of_variables=DTLZ_N_VAR["dtlz6"], number_of_objectives=DTLZ_N_OBJ),
                "dtlz7": DTLZ7(number_of_variables=DTLZ_N_VAR["dtlz7"], number_of_objectives=DTLZ_N_OBJ),
                "wfg1": WFG1(number_of_variables=WFG_N_VAR, number_of_objectives=WFG_N_OBJ, k=wfg_k, l=wfg_l),
                "wfg2": WFG2(number_of_variables=WFG_N_VAR, number_of_objectives=WFG_N_OBJ, k=wfg_k, l=wfg_l),
                "wfg3": WFG3(number_of_variables=WFG_N_VAR, number_of_objectives=WFG_N_OBJ, k=wfg_k, l=wfg_l),
                "wfg4": WFG4(number_of_variables=WFG_N_VAR, number_of_objectives=WFG_N_OBJ, k=wfg_k, l=wfg_l),
                "wfg5": WFG5(number_of_variables=WFG_N_VAR, number_of_objectives=WFG_N_OBJ, k=wfg_k, l=wfg_l),
                "wfg6": WFG6(number_of_variables=WFG_N_VAR, number_of_objectives=WFG_N_OBJ, k=wfg_k, l=wfg_l),
                "wfg7": WFG7(number_of_variables=WFG_N_VAR, number_of_objectives=WFG_N_OBJ, k=wfg_k, l=wfg_l),
                "wfg8": WFG8(number_of_variables=WFG_N_VAR, number_of_objectives=WFG_N_OBJ, k=wfg_k, l=wfg_l),
                "wfg9": WFG9(number_of_variables=WFG_N_VAR, number_of_objectives=WFG_N_OBJ, k=wfg_k, l=wfg_l),
            }
        except Exception as e:  # pragma: no cover
            print(f"  Warning: jMetalPy alignment check skipped (import failed): {e}")
            check_jmetal = False

    platypus_problem_map = None
    if check_platypus:
        try:
            from platypus import Problem, Real, Solution

            class PlatypusVAMOSProblem(Problem):
                def __init__(self, name: str, n_var: int, n_obj: int):
                    selection = make_problem_selection(name, n_var=n_var, n_obj=n_obj)
                    self._problem = selection.instantiate()
                    super().__init__(self._problem.n_var, self._problem.n_obj)
                    xl, xu = _as_1d_bounds(self._problem.xl, self._problem.xu, self._problem.n_var)
                    self.types[:] = [Real(lo, hi) for lo, hi in zip(xl, xu)]

                def evaluate(self, solution):
                    X = np.asarray(solution.variables, dtype=float).reshape(1, -1)
                    out = {"F": np.zeros((1, self._problem.n_obj), dtype=float)}
                    self._problem.evaluate(X, out)
                    solution.objectives[:] = out["F"][0]

            class PlatypusWFG(Problem):
                def __init__(self, wfg_num: int, n_var: int, n_obj: int):
                    super().__init__(n_var, n_obj)
                    self.types[:] = [Real(0, 2 * (i + 1)) for i in range(n_var)]
                    from pymoo.problems import get_problem

                    self._pymoo_problem = get_problem(f"wfg{wfg_num}", n_var=n_var, n_obj=n_obj)

                def evaluate(self, solution):
                    x = np.array(solution.variables, dtype=float)
                    out = {"F": None}
                    self._pymoo_problem._evaluate(x.reshape(1, -1), out)
                    solution.objectives[:] = out["F"][0]

            platypus_problem_map = {
                "zdt1": PlatypusVAMOSProblem("zdt1", n_var=ZDT_N_VAR["zdt1"], n_obj=ZDT_N_OBJ),
                "zdt2": PlatypusVAMOSProblem("zdt2", n_var=ZDT_N_VAR["zdt2"], n_obj=ZDT_N_OBJ),
                "zdt3": PlatypusVAMOSProblem("zdt3", n_var=ZDT_N_VAR["zdt3"], n_obj=ZDT_N_OBJ),
                "zdt4": PlatypusVAMOSProblem("zdt4", n_var=ZDT_N_VAR["zdt4"], n_obj=ZDT_N_OBJ),
                "zdt6": PlatypusVAMOSProblem("zdt6", n_var=ZDT_N_VAR["zdt6"], n_obj=ZDT_N_OBJ),
                "dtlz1": PlatypusVAMOSProblem("dtlz1", n_var=DTLZ_N_VAR["dtlz1"], n_obj=DTLZ_N_OBJ),
                "dtlz2": PlatypusVAMOSProblem("dtlz2", n_var=DTLZ_N_VAR["dtlz2"], n_obj=DTLZ_N_OBJ),
                "dtlz3": PlatypusVAMOSProblem("dtlz3", n_var=DTLZ_N_VAR["dtlz3"], n_obj=DTLZ_N_OBJ),
                "dtlz4": PlatypusVAMOSProblem("dtlz4", n_var=DTLZ_N_VAR["dtlz4"], n_obj=DTLZ_N_OBJ),
                "dtlz5": PlatypusVAMOSProblem("dtlz5", n_var=DTLZ_N_VAR["dtlz5"], n_obj=DTLZ_N_OBJ),
                "dtlz6": PlatypusVAMOSProblem("dtlz6", n_var=DTLZ_N_VAR["dtlz6"], n_obj=DTLZ_N_OBJ),
                "dtlz7": PlatypusVAMOSProblem("dtlz7", n_var=DTLZ_N_VAR["dtlz7"], n_obj=DTLZ_N_OBJ),
                "wfg1": PlatypusWFG(1, n_var=WFG_N_VAR, n_obj=WFG_N_OBJ),
                "wfg2": PlatypusWFG(2, n_var=WFG_N_VAR, n_obj=WFG_N_OBJ),
                "wfg3": PlatypusWFG(3, n_var=WFG_N_VAR, n_obj=WFG_N_OBJ),
                "wfg4": PlatypusWFG(4, n_var=WFG_N_VAR, n_obj=WFG_N_OBJ),
                "wfg5": PlatypusWFG(5, n_var=WFG_N_VAR, n_obj=WFG_N_OBJ),
                "wfg6": PlatypusWFG(6, n_var=WFG_N_VAR, n_obj=WFG_N_OBJ),
                "wfg7": PlatypusWFG(7, n_var=WFG_N_VAR, n_obj=WFG_N_OBJ),
                "wfg8": PlatypusWFG(8, n_var=WFG_N_VAR, n_obj=WFG_N_OBJ),
                "wfg9": PlatypusWFG(9, n_var=WFG_N_VAR, n_obj=WFG_N_OBJ),
            }
        except Exception as e:  # pragma: no cover
            print(f"  Warning: Platypus alignment check skipped (import failed): {e}")
            check_platypus = False

    checked = 0
    for problem_name in PROBLEMS:
        n_var, n_obj = problem_dims(problem_name)
        selection = make_problem_selection(problem_name, n_var=n_var, n_obj=n_obj)
        ref_problem = selection.instantiate()
        xl, xu = _as_1d_bounds(ref_problem.xl, ref_problem.xu, n_var)
        X = rng.uniform(xl, xu, size=(n_samples, n_var))

        out = {"F": np.zeros((n_samples, n_obj), dtype=float)}
        ref_problem.evaluate(X, out)
        F_ref = out["F"]

        def _check(name: str, F_other: np.ndarray) -> None:
            nonlocal checked
            checked += 1
            if not np.allclose(F_ref, F_other, rtol=ALIGN_CHECK_RTOL, atol=ALIGN_CHECK_ATOL):
                diff = np.abs(F_ref - F_other)
                max_abs = float(np.max(diff))
                raise RuntimeError(f"Objective mismatch for {problem_name} vs {name}: max|Δ|={max_abs:.3e}")

        if check_pymoo and pymoo_get_problem is not None:
            if problem_name.startswith("zdt"):
                pymoo_problem = pymoo_get_problem(problem_name, n_var=n_var)
            else:
                pymoo_problem = pymoo_get_problem(problem_name, n_var=n_var, n_obj=n_obj)
            out_p = {"F": None}
            pymoo_problem._evaluate(X, out_p)
            _check("pymoo", np.asarray(out_p["F"], dtype=float))

        if check_jmetal and jmetal_problem_map is not None:
            jm = jmetal_problem_map[problem_name]
            F = []
            for x in X:
                sol = jm.create_solution()
                sol.variables = [float(v) for v in x]
                jm.evaluate(sol)
                F.append(sol.objectives)
            _check("jMetalPy", np.asarray(F, dtype=float))

        if check_platypus and platypus_problem_map is not None:
            from platypus import Solution  # type: ignore[import-not-found]

            pp = platypus_problem_map[problem_name]
            F = []
            for x in X:
                sol = Solution(pp)
                sol.variables[:] = [float(v) for v in x]
                pp.evaluate(sol)
                F.append(sol.objectives)
            _check("Platypus", np.asarray(F, dtype=float))

    print(f"Objective alignment check passed ({checked} comparisons)")


# =============================================================================
# DEAP PROBLEM IMPLEMENTATIONS (using VAMOS definitions)


def _resolve_bounds(problem, n_var: int) -> tuple[np.ndarray, np.ndarray]:
    xl = np.asarray(problem.xl, dtype=float)
    xu = np.asarray(problem.xu, dtype=float)
    if xl.ndim == 0:
        xl = np.full(n_var, float(xl))
    if xu.ndim == 0:
        xu = np.full(n_var, float(xu))
    if xl.shape != (n_var,) or xu.shape != (n_var,):
        raise ValueError(f"Invalid bounds for {problem.__class__.__name__}: xl={xl.shape}, xu={xu.shape}")
    return xl, xu


def get_deap_problem(problem_name: str, n_var: int, n_obj: int):
    """Get DEAP-compatible problem function and bounds using VAMOS definitions."""
    problem = make_problem_selection(problem_name, n_var=n_var, n_obj=n_obj).instantiate()
    xl, xu = _resolve_bounds(problem, n_var)

    def evaluate(individual):
        X = np.asarray(individual, dtype=float).reshape(1, -1)
        out = {"F": np.zeros((1, n_obj), dtype=float)}
        problem.evaluate(X, out)
        return tuple(out["F"][0])

    return evaluate, xl, xu


def run_single_benchmark(problem_name, seed, framework):
    """Run a single benchmark configuration."""
    result_entry = None
    n_var, n_obj = problem_dims(problem_name)

    # VAMOS backends
    if framework.startswith("vamos-"):
        backend = framework.replace("vamos-", "")
        try:
            problem = make_problem_selection(problem_name, n_var=n_var, n_obj=n_obj).instantiate()
            algo_config = (
                NSGAIIConfig.builder()
                .pop_size(POP_SIZE)
                .crossover("sbx", prob=CROSSOVER_PROB, eta=CROSSOVER_ETA)
                .mutation("pm", prob=1.0 / n_var, eta=MUTATION_ETA)
                .selection("tournament")
                .build()
            )

            if backend == "numba" and NUMBA_WARMUP_EVALS > 0:
                warmup_budget = min(int(NUMBA_WARMUP_EVALS), int(N_EVALS))
                _ = optimize(
                    problem,
                    algorithm="nsgaii",
                    algorithm_config=algo_config,
                    termination=("n_eval", warmup_budget),
                    seed=seed,
                    engine=backend,
                )

            start = time.perf_counter()
            result = optimize(
                problem,
                algorithm="nsgaii",
                algorithm_config=algo_config,
                termination=("n_eval", N_EVALS),
                seed=seed,
                engine=backend,
            )
            elapsed = time.perf_counter() - start
            hv = compute_hv(result.F, problem_name) if result.F is not None else float("nan")
            result_entry = {
                "framework": f"VAMOS ({backend})",
                "problem": problem_name,
                "algorithm": "NSGA-II",
                "n_evals": N_EVALS,
                "seed": seed,
                "runtime_seconds": elapsed,
                "n_solutions": result.X.shape[0] if result.X is not None else 0,
                "hypervolume": hv,
            }
            print(f"  {problem_name} VAMOS({backend}) seed={seed}: {elapsed:.2f}s")
        except Exception as e:
            print(f"  {problem_name} VAMOS({backend}) seed={seed} FAILED: {e}")

    # pymoo
    elif framework == "pymoo":
        try:
            from pymoo.algorithms.moo.nsga2 import NSGA2
            from pymoo.optimize import minimize
            from pymoo.termination import get_termination
            from pymoo.problems import get_problem
            from pymoo.operators.crossover.sbx import SBX
            from pymoo.operators.mutation.pm import PM

            if problem_name.startswith("zdt"):
                pymoo_problem = get_problem(problem_name, n_var=n_var)
            else:
                pymoo_problem = get_problem(problem_name, n_var=n_var, n_obj=n_obj)

            algorithm = NSGA2(
                pop_size=POP_SIZE,
                crossover=SBX(prob=CROSSOVER_PROB, eta=CROSSOVER_ETA),
                mutation=PM(prob=1.0, prob_var=1.0 / n_var, eta=MUTATION_ETA),
            )
            termination = get_termination("n_eval", N_EVALS)

            start = time.perf_counter()
            res = minimize(pymoo_problem, algorithm, termination, seed=seed, verbose=False)
            elapsed = time.perf_counter() - start
            hv = compute_hv(res.F, problem_name) if res.F is not None else float("nan")
            result_entry = {
                "framework": "pymoo",
                "problem": problem_name,
                "algorithm": "NSGA-II",
                "n_evals": N_EVALS,
                "seed": seed,
                "runtime_seconds": elapsed,
                "n_solutions": res.X.shape[0] if res.X is not None else 0,
                "hypervolume": hv,
            }
            print(f"  {problem_name} pymoo seed={seed}: {elapsed:.2f}s")
        except Exception as e:
            print(f"  {problem_name} pymoo seed={seed} FAILED: {e}")

    # DEAP
    elif framework == "deap":
        try:
            from deap import base, creator, tools
            import copy
            import random

            problem_func, xl, xu = get_deap_problem(problem_name, n_var=n_var, n_obj=n_obj)
            xl_list = xl.tolist()
            xu_list = xu.tolist()

            # Setup DEAP (use per-objective fitness/individual to avoid shape mismatches)
            fitness_name = f"FitnessMin{n_obj}"
            individual_name = f"Individual{n_obj}"
            if not hasattr(creator, fitness_name):
                creator.create(fitness_name, base.Fitness, weights=(-1.0,) * n_obj)
            if not hasattr(creator, individual_name):
                creator.create(individual_name, list, fitness=getattr(creator, fitness_name))

            toolbox = base.Toolbox()

            def _random_individual():
                return [random.uniform(lo, hi) for lo, hi in zip(xl_list, xu_list)]

            toolbox.register("individual", tools.initIterate, getattr(creator, individual_name), _random_individual)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", problem_func)
            toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=xl_list, up=xu_list, eta=CROSSOVER_ETA)
            toolbox.register("mutate", tools.mutPolynomialBounded, low=xl_list, up=xu_list, eta=MUTATION_ETA, indpb=1.0 / n_var)
            toolbox.register("select", tools.selNSGA2)
            toolbox.register("select_tournament", tools.selTournamentDCD)
            toolbox.register("clone", copy.deepcopy)

            random.seed(seed)
            pop = toolbox.population(n=POP_SIZE)
            n_gen = max(0, (N_EVALS - POP_SIZE) // POP_SIZE)

            start = time.perf_counter()

            # Evaluate the initial population
            invalid = [ind for ind in pop if not ind.fitness.valid]
            for ind in invalid:
                ind.fitness.values = toolbox.evaluate(ind)

            # Assign crowding distance
            pop = toolbox.select(pop, len(pop))

            for _ in range(n_gen):
                offspring = toolbox.select_tournament(pop, len(pop))
                offspring = list(map(toolbox.clone, offspring))

                for i in range(0, len(offspring), 2):
                    if random.random() <= CROSSOVER_PROB:
                        toolbox.mate(offspring[i], offspring[i + 1])
                        del offspring[i].fitness.values
                        del offspring[i + 1].fitness.values

                for mutant in offspring:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

                invalid = [ind for ind in offspring if not ind.fitness.valid]
                for ind in invalid:
                    ind.fitness.values = toolbox.evaluate(ind)

                pop = toolbox.select(pop + offspring, POP_SIZE)

            elapsed = time.perf_counter() - start

            fronts = tools.sortNondominated(pop, len(pop), first_front_only=True)
            F = np.array([ind.fitness.values for ind in fronts[0]])
            hv = compute_hv(F, problem_name)

            result_entry = {
                "framework": "DEAP",
                "problem": problem_name,
                "algorithm": "NSGA-II",
                "n_evals": N_EVALS,
                "seed": seed,
                "runtime_seconds": elapsed,
                "n_solutions": len(fronts[0]),
                "hypervolume": hv,
            }
            print(f"  {problem_name} DEAP seed={seed}: {elapsed:.2f}s")
        except Exception as e:
            print(f"  {problem_name} DEAP seed={seed} FAILED: {e}")

    # jMetalPy
    elif framework == "jmetalpy":
        try:
            from jmetal.algorithm.multiobjective import NSGAII
            from jmetal.operator.crossover import SBXCrossover
            from jmetal.operator.mutation import PolynomialMutation
            from jmetal.util.termination_criterion import StoppingByEvaluations
            from jmetal.problem import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
            from jmetal.problem import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7
            from jmetal.problem import WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9
            import random

            random.seed(seed)
            np.random.seed(seed)

            # Align WFG parameterization with VAMOS/pymoo defaults (k=4 for m=2).
            wfg_k = 4 if WFG_N_OBJ == 2 else 2 * (WFG_N_OBJ - 1)
            wfg_l = WFG_N_VAR - wfg_k

            problem_map = {
                "zdt1": ZDT1(number_of_variables=ZDT_N_VAR["zdt1"]),
                "zdt2": ZDT2(number_of_variables=ZDT_N_VAR["zdt2"]),
                "zdt3": ZDT3(number_of_variables=ZDT_N_VAR["zdt3"]),
                "zdt4": ZDT4(number_of_variables=ZDT_N_VAR["zdt4"]),
                "zdt6": ZDT6(number_of_variables=ZDT_N_VAR["zdt6"]),
                "dtlz1": DTLZ1(number_of_variables=DTLZ_N_VAR["dtlz1"], number_of_objectives=DTLZ_N_OBJ),
                "dtlz2": DTLZ2(number_of_variables=DTLZ_N_VAR["dtlz2"], number_of_objectives=DTLZ_N_OBJ),
                "dtlz3": DTLZ3(number_of_variables=DTLZ_N_VAR["dtlz3"], number_of_objectives=DTLZ_N_OBJ),
                "dtlz4": DTLZ4(number_of_variables=DTLZ_N_VAR["dtlz4"], number_of_objectives=DTLZ_N_OBJ),
                "dtlz5": DTLZ5(number_of_variables=DTLZ_N_VAR["dtlz5"], number_of_objectives=DTLZ_N_OBJ),
                "dtlz6": DTLZ6(number_of_variables=DTLZ_N_VAR["dtlz6"], number_of_objectives=DTLZ_N_OBJ),
                "dtlz7": DTLZ7(number_of_variables=DTLZ_N_VAR["dtlz7"], number_of_objectives=DTLZ_N_OBJ),
                "wfg1": WFG1(number_of_variables=WFG_N_VAR, number_of_objectives=WFG_N_OBJ, k=wfg_k, l=wfg_l),
                "wfg2": WFG2(number_of_variables=WFG_N_VAR, number_of_objectives=WFG_N_OBJ, k=wfg_k, l=wfg_l),
                "wfg3": WFG3(number_of_variables=WFG_N_VAR, number_of_objectives=WFG_N_OBJ, k=wfg_k, l=wfg_l),
                "wfg4": WFG4(number_of_variables=WFG_N_VAR, number_of_objectives=WFG_N_OBJ, k=wfg_k, l=wfg_l),
                "wfg5": WFG5(number_of_variables=WFG_N_VAR, number_of_objectives=WFG_N_OBJ, k=wfg_k, l=wfg_l),
                "wfg6": WFG6(number_of_variables=WFG_N_VAR, number_of_objectives=WFG_N_OBJ, k=wfg_k, l=wfg_l),
                "wfg7": WFG7(number_of_variables=WFG_N_VAR, number_of_objectives=WFG_N_OBJ, k=wfg_k, l=wfg_l),
                "wfg8": WFG8(number_of_variables=WFG_N_VAR, number_of_objectives=WFG_N_OBJ, k=wfg_k, l=wfg_l),
                "wfg9": WFG9(number_of_variables=WFG_N_VAR, number_of_objectives=WFG_N_OBJ, k=wfg_k, l=wfg_l),
            }

            if problem_name not in problem_map:
                raise ValueError(f"Problem {problem_name} not available in jMetalPy")

            jmetal_problem = problem_map[problem_name]

            algorithm = NSGAII(
                problem=jmetal_problem,
                population_size=POP_SIZE,
                offspring_population_size=POP_SIZE,
                mutation=PolynomialMutation(probability=1.0 / n_var, distribution_index=MUTATION_ETA),
                crossover=SBXCrossover(probability=CROSSOVER_PROB, distribution_index=CROSSOVER_ETA),
                termination_criterion=StoppingByEvaluations(max_evaluations=N_EVALS),
            )

            start = time.perf_counter()
            algorithm.run()
            elapsed = time.perf_counter() - start

            solutions = algorithm.result()  # result() is a method, not property
            F = np.array([s.objectives for s in solutions])
            hv = compute_hv(F, problem_name)

            result_entry = {
                "framework": "jMetalPy",
                "problem": problem_name,
                "algorithm": "NSGA-II",
                "n_evals": N_EVALS,
                "seed": seed,
                "runtime_seconds": elapsed,
                "n_solutions": len(solutions),
                "hypervolume": hv,
            }
            print(f"  {problem_name} jMetalPy seed={seed}: {elapsed:.2f}s")
        except Exception as e:
            print(f"  {problem_name} jMetalPy seed={seed} FAILED: {e}")

    # Platypus
    elif framework == "platypus":
        try:
            from platypus import NSGAII as PlatypusNSGAII, Problem, Real
            from platypus import GAOperator, PM, SBX, TournamentSelector
            import random

            random.seed(seed)

            # Custom VAMOS wrapper for Platypus (aligns bounds/definitions)
            class PlatypusVAMOSProblem(Problem):
                def __init__(self, name: str, n_var: int, n_obj: int):
                    selection = make_problem_selection(name, n_var=n_var, n_obj=n_obj)
                    self._problem = selection.instantiate()
                    super().__init__(self._problem.n_var, self._problem.n_obj)
                    xl = np.asarray(self._problem.xl, dtype=float)
                    xu = np.asarray(self._problem.xu, dtype=float)
                    if xl.ndim == 0:
                        xl = np.full(self._problem.n_var, float(xl))
                    if xu.ndim == 0:
                        xu = np.full(self._problem.n_var, float(xu))
                    if xl.shape != (self._problem.n_var,) or xu.shape != (self._problem.n_var,):
                        raise ValueError(f"Invalid bounds for {name}: xl={xl.shape}, xu={xu.shape}")
                    self.types[:] = [Real(lo, hi) for lo, hi in zip(xl, xu)]

                def evaluate(self, solution):
                    X = np.asarray(solution.variables, dtype=float).reshape(1, -1)
                    out = {"F": np.zeros((1, self._problem.n_obj), dtype=float)}
                    self._problem.evaluate(X, out)
                    solution.objectives[:] = out["F"][0]

            # Custom WFG wrapper for Platypus (using pymoo as backend)
            class PlatypusWFG(Problem):
                def __init__(self, wfg_num, n_var=24, n_obj=2):
                    super().__init__(n_var, n_obj)
                    self.types[:] = [Real(0, 2 * (i + 1)) for i in range(n_var)]
                    from pymoo.problems import get_problem

                    self._pymoo_problem = get_problem(f"wfg{wfg_num}", n_var=n_var, n_obj=n_obj)

                def evaluate(self, solution):
                    x = np.array(solution.variables)
                    out = {"F": None}
                    self._pymoo_problem._evaluate(x.reshape(1, -1), out)
                    solution.objectives[:] = out["F"][0]

            problem_map = {
                "zdt1": PlatypusVAMOSProblem("zdt1", n_var=ZDT_N_VAR["zdt1"], n_obj=ZDT_N_OBJ),
                "zdt2": PlatypusVAMOSProblem("zdt2", n_var=ZDT_N_VAR["zdt2"], n_obj=ZDT_N_OBJ),
                "zdt3": PlatypusVAMOSProblem("zdt3", n_var=ZDT_N_VAR["zdt3"], n_obj=ZDT_N_OBJ),
                "zdt4": PlatypusVAMOSProblem("zdt4", n_var=ZDT_N_VAR["zdt4"], n_obj=ZDT_N_OBJ),
                "zdt6": PlatypusVAMOSProblem("zdt6", n_var=ZDT_N_VAR["zdt6"], n_obj=ZDT_N_OBJ),
                "dtlz1": PlatypusVAMOSProblem("dtlz1", n_var=DTLZ_N_VAR["dtlz1"], n_obj=DTLZ_N_OBJ),
                "dtlz2": PlatypusVAMOSProblem("dtlz2", n_var=DTLZ_N_VAR["dtlz2"], n_obj=DTLZ_N_OBJ),
                "dtlz3": PlatypusVAMOSProblem("dtlz3", n_var=DTLZ_N_VAR["dtlz3"], n_obj=DTLZ_N_OBJ),
                "dtlz4": PlatypusVAMOSProblem("dtlz4", n_var=DTLZ_N_VAR["dtlz4"], n_obj=DTLZ_N_OBJ),
                "dtlz5": PlatypusVAMOSProblem("dtlz5", n_var=DTLZ_N_VAR["dtlz5"], n_obj=DTLZ_N_OBJ),
                "dtlz6": PlatypusVAMOSProblem("dtlz6", n_var=DTLZ_N_VAR["dtlz6"], n_obj=DTLZ_N_OBJ),
                "dtlz7": PlatypusVAMOSProblem("dtlz7", n_var=DTLZ_N_VAR["dtlz7"], n_obj=DTLZ_N_OBJ),
                "wfg1": PlatypusWFG(1, n_var=WFG_N_VAR, n_obj=WFG_N_OBJ),
                "wfg2": PlatypusWFG(2, n_var=WFG_N_VAR, n_obj=WFG_N_OBJ),
                "wfg3": PlatypusWFG(3, n_var=WFG_N_VAR, n_obj=WFG_N_OBJ),
                "wfg4": PlatypusWFG(4, n_var=WFG_N_VAR, n_obj=WFG_N_OBJ),
                "wfg5": PlatypusWFG(5, n_var=WFG_N_VAR, n_obj=WFG_N_OBJ),
                "wfg6": PlatypusWFG(6, n_var=WFG_N_VAR, n_obj=WFG_N_OBJ),
                "wfg7": PlatypusWFG(7, n_var=WFG_N_VAR, n_obj=WFG_N_OBJ),
                "wfg8": PlatypusWFG(8, n_var=WFG_N_VAR, n_obj=WFG_N_OBJ),
                "wfg9": PlatypusWFG(9, n_var=WFG_N_VAR, n_obj=WFG_N_OBJ),
            }

            if problem_name not in problem_map:
                raise ValueError(f"Problem {problem_name} not available in Platypus")

            platypus_problem = problem_map[problem_name]

            variator = GAOperator(
                SBX(probability=CROSSOVER_PROB, distribution_index=CROSSOVER_ETA),
                PM(probability=1.0 / n_var, distribution_index=MUTATION_ETA),
            )
            algorithm = PlatypusNSGAII(
                platypus_problem,
                population_size=POP_SIZE,
                selector=TournamentSelector(2),
                variator=variator,
            )

            start = time.perf_counter()
            algorithm.run(N_EVALS)
            elapsed = time.perf_counter() - start

            F = np.array([s.objectives for s in algorithm.result])
            hv = compute_hv(F, problem_name)

            result_entry = {
                "framework": "Platypus",
                "problem": problem_name,
                "algorithm": "NSGA-II",
                "n_evals": N_EVALS,
                "seed": seed,
                "runtime_seconds": elapsed,
                "n_solutions": len(algorithm.result),
                "hypervolume": hv,
            }
            print(f"  {problem_name} Platypus seed={seed}: {elapsed:.2f}s")
        except Exception as e:
            print(f"  {problem_name} Platypus seed={seed} FAILED: {e}")

    return result_entry


# Preflight objective alignment (guards against definition drift)
run_objective_alignment_checks()

# Build list of all jobs - split by thread-safety
PARALLEL_FRAMEWORKS = ["vamos-numpy", "vamos-numba", "vamos-moocore", "pymoo", "deap", "jmetalpy", "platypus"]
SEQUENTIAL_FRAMEWORKS = []

parallel_jobs = []
sequential_jobs = []

for problem_name in PROBLEMS:
    for seed in range(N_SEEDS):
        for framework in FRAMEWORKS:
            job = (problem_name, seed, framework)
            if framework in SEQUENTIAL_FRAMEWORKS:
                sequential_jobs.append(job)
            else:
                parallel_jobs.append(job)

print(f"\nParallel jobs: {len(parallel_jobs)}")
print(f"Sequential jobs: {len(sequential_jobs)}")
print(f"Total: {len(parallel_jobs) + len(sequential_jobs)}")

# Run parallel jobs first
print(f"\nRunning {len(parallel_jobs)} parallel jobs...")
if parallel_jobs:
    if N_JOBS <= 1:
        bar = ProgressBar(total=len(parallel_jobs), desc="Paper benchmark")
        results_list = []
        for p, s, b in parallel_jobs:
            results_list.append(run_single_benchmark(p, s, b))
            bar.update(1)
        bar.close()
    else:
        with joblib_progress(total=len(parallel_jobs), desc="Paper benchmark"):
            results_list = Parallel(n_jobs=N_JOBS)(delayed(run_single_benchmark)(p, s, b) for p, s, b in parallel_jobs)
else:
    results_list = []

# Run sequential jobs (jMetalPy, Platypus)
print(f"\nRunning {len(sequential_jobs)} sequential jobs...")
seq_bar = ProgressBar(total=len(sequential_jobs), desc="Sequential jobs") if sequential_jobs else None
for p, s, b in sequential_jobs:
    result = run_single_benchmark(p, s, b)
    if result:
        results_list.append(result)
    if seq_bar is not None:
        seq_bar.update(1)
if seq_bar is not None:
    seq_bar.close()

# Filter out None results (failed runs)
results = [r for r in results_list if r is not None]

# Save results
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(df)} results to {OUTPUT_CSV}")
print("\nDone!")
