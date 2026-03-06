"""
VAMOS Paper ZCAT Benchmark for All Tables
==========================================
Runs ZCAT benchmarks for every table configuration in the paper:
  - NSGA-II generational (all backends + all frameworks)
  - NSGA-II steady-state (VAMOS + pymoo + jMetalPy)
  - NSGA-II ext. archive (VAMOS + pymoo + jMetalPy)
  - SMS-EMOA (VAMOS + pymoo + jMetalPy)
  - MOEA/D (VAMOS + pymoo + jMetalPy + Platypus)

All frameworks evaluate the same VAMOS ZCAT problem definitions (2 objectives,
30 variables) to ensure cross-framework alignment by construction.

Results are appended to the correct per-algorithm CSV so that existing table
generation scripts (04, 05, 14, 16, 17) can pick up ZCAT data automatically.

Usage: python paper/31_run_zcat_all_tables.py

Environment variables:
  VAMOS_N_EVALS              evaluations per run (default: 50000)
  VAMOS_N_SEEDS              independent seeds (default: 30)
  VAMOS_N_JOBS               parallel workers (default: cpu_count - 1)
  VAMOS_ZCAT_PROBLEMS        comma-separated problem IDs 1-20 (default: all)
  VAMOS_ZCAT_ALGORITHMS      comma-separated algorithms (default: all)
                             Options: nsgaii, nsgaii_ss, nsgaii_archive, smsemoa, moead
  VAMOS_ZCAT_FRAMEWORKS      comma-separated (overrides per-algorithm defaults)
  VAMOS_NUMBA_WARMUP_EVALS   Numba warmup evals (default: 2000)
  VAMOS_ZCAT_SAVE_EVERY      chunk size for partial saves (default: 50)
  VAMOS_ZCAT_RESUME          1/0 resume from existing CSV (default: 1)
  VAMOS_DEAP_N_JOBS          DEAP pool workers (default: 1; Windows-safe)
"""

from __future__ import annotations

import copy
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from vamos import optimize
from vamos.engine.algorithm.config import MOEADConfig, NSGAIIConfig, SMSEMOAConfig
from vamos.engine.algorithm.components.subset_selection import select_top_k_crowding
from vamos.foundation.problem.registry import make_problem_selection
from vamos.resources import weight_path

try:
    from .benchmark_utils import compute_hv, compute_igd_plus
except ImportError:
    from benchmark_utils import compute_hv, compute_igd_plus

try:
    from .progress_utils import ProgressBar, joblib_progress
except ImportError:  # pragma: no cover
    from progress_utils import ProgressBar, joblib_progress

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = ROOT_DIR / "experiments"


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None else int(raw)


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_str_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _unique_preserve_order(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


N_EVALS = _int_env("VAMOS_N_EVALS", 50_000)
N_SEEDS = _int_env("VAMOS_N_SEEDS", 30)
N_JOBS = _int_env("VAMOS_N_JOBS", max(1, (os.cpu_count() or 2) - 1))
NUMBA_WARMUP_EVALS = _int_env("VAMOS_NUMBA_WARMUP_EVALS", 2000)
SAVE_EVERY = _int_env("VAMOS_ZCAT_SAVE_EVERY", 50)
RESUME = os.environ.get("VAMOS_ZCAT_RESUME", "1").strip().lower() not in {"0", "false", "no"}
DEAP_N_JOBS = _int_env("VAMOS_DEAP_N_JOBS", 1)
if os.name == "nt" and DEAP_N_JOBS > 1:
    print("Info: forcing DEAP workers to 1 on Windows (spawn mode cannot pickle local evaluator).")
    DEAP_N_JOBS = 1

# ZCAT problem configuration: 2 objectives, 30 variables (standard)
N_OBJ = 2
N_VAR = 30

_problems_raw = os.environ.get("VAMOS_ZCAT_PROBLEMS", ",".join(str(i) for i in range(1, 21)))
PROBLEM_IDS = _parse_int_list(_problems_raw)

# Operator parameters (same as 01_run_paper_benchmark.py)
POP_SIZE = 100
CROSSOVER_PROB = 1.0
CROSSOVER_ETA = 20.0
MUTATION_ETA = 20.0
MOEAD_NEIGHBOR_SIZE = 20
MOEAD_DELTA = 0.9
MOEAD_REPLACE_LIMIT = 2
MOEAD_WEIGHTS_DIR = weight_path("W3D_100.dat").parent

TOP_K_HV = POP_SIZE  # trim archive to pop_size best-spread solutions for HV/IGD+

# Algorithm → CSV mapping (same as script 01)
ALGO_CSV_MAP = {
    "nsgaii": DATA_DIR / "benchmark_paper.csv",
    "nsgaii_ss": DATA_DIR / "benchmark_paper_nsgaii_ss.csv",
    "nsgaii_archive": DATA_DIR / "benchmark_paper_nsgaii_archive.csv",
    "smsemoa": DATA_DIR / "benchmark_paper_smsemoa.csv",
    "moead": DATA_DIR / "benchmark_paper_moead.csv",
}

ALGO_DISPLAY = {
    "nsgaii": "NSGA-II",
    "nsgaii_ss": "NSGA-II (steady-state)",
    "nsgaii_archive": "NSGA-II (ext. archive)",
    "smsemoa": "SMS-EMOA",
    "moead": "MOEA/D",
}

# Framework availability per algorithm (matches script 01)
ALGO_FRAMEWORKS = {
    "nsgaii": ["vamos-numba", "vamos-numpy", "vamos-moocore", "pymoo", "jmetalpy", "deap", "platypus", "pygmo"],
    "nsgaii_ss": ["vamos-numba", "pymoo", "jmetalpy"],
    "nsgaii_archive": ["vamos-numba", "pymoo", "jmetalpy"],
    "smsemoa": ["vamos-numba", "pymoo", "jmetalpy"],
    "moead": ["vamos-numba", "pymoo", "jmetalpy", "platypus"],
}

# Parse algorithms to run
_algo_raw = os.environ.get("VAMOS_ZCAT_ALGORITHMS", "nsgaii,nsgaii_ss,nsgaii_archive,smsemoa,moead")
ALGORITHMS = _parse_str_list(_algo_raw)

# Optional framework override
_fw_override = os.environ.get("VAMOS_ZCAT_FRAMEWORKS")


# =============================================================================
# HELPERS
# =============================================================================


def _trim_for_hv(F: np.ndarray | None, k: int = TOP_K_HV) -> np.ndarray | None:
    """Trim a non-dominated front to the *k* most spread solutions."""
    if F is None or F.shape[0] <= k:
        return F
    return F[select_top_k_crowding(F, k)]


def _make_vamos_zcat(pid: int, n_var: int, n_obj: int):
    """Instantiate a VAMOS ZCAT problem."""
    return make_problem_selection(f"zcat{pid}", n_var=n_var, n_obj=n_obj).instantiate()


def _make_pymoo_zcat(pid: int, n_var: int, n_obj: int):
    """Create a pymoo Problem wrapping VAMOS ZCAT evaluation."""
    from pymoo.core.problem import Problem as PymooProblem

    vamos_prob = _make_vamos_zcat(pid, n_var, n_obj)
    xl = np.asarray(vamos_prob.xl, dtype=float)
    xu = np.asarray(vamos_prob.xu, dtype=float)

    class _ZCATWrapper(PymooProblem):
        def __init__(self):
            super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)
            self._vamos = vamos_prob

        def _evaluate(self, x, out, *args, **kwargs):
            F = np.zeros((x.shape[0], n_obj), dtype=float)
            o = {"F": F}
            self._vamos.evaluate(x, o)
            out["F"] = o["F"]

    return _ZCATWrapper()


def _make_jmetal_zcat(pid: int, n_var: int, n_obj: int):
    """Create a jMetalPy FloatProblem wrapping VAMOS ZCAT evaluation."""
    from jmetal.core.problem import FloatProblem

    vamos_prob = _make_vamos_zcat(pid, n_var, n_obj)
    xl = np.asarray(vamos_prob.xl, dtype=float).tolist()
    xu = np.asarray(vamos_prob.xu, dtype=float).tolist()

    class _ZCATWrapper(FloatProblem):
        def __init__(self):
            super().__init__()
            self.lower_bound = xl
            self.upper_bound = xu
            self._vamos = vamos_prob

        def name(self) -> str:
            return f"ZCAT{pid}"

        def number_of_objectives(self) -> int:
            return n_obj

        def number_of_constraints(self) -> int:
            return 0

        def number_of_variables(self) -> int:
            return n_var

        def evaluate(self, solution):
            x = np.asarray(solution.variables, dtype=float).reshape(1, -1)
            out = {"F": np.zeros((1, n_obj), dtype=float)}
            self._vamos.evaluate(x, out)
            solution.objectives = out["F"][0].tolist()
            return solution

    return _ZCATWrapper()


def _make_deap_zcat(pid: int, n_var: int, n_obj: int):
    """Return (evaluate_fn, xl, xu) for DEAP wrapping VAMOS ZCAT."""
    vamos_prob = _make_vamos_zcat(pid, n_var, n_obj)
    xl = np.asarray(vamos_prob.xl, dtype=float)
    xu = np.asarray(vamos_prob.xu, dtype=float)

    def evaluate(individual):
        X = np.asarray(individual, dtype=float).reshape(1, -1)
        out = {"F": np.zeros((1, n_obj), dtype=float)}
        vamos_prob.evaluate(X, out)
        return tuple(out["F"][0])

    return evaluate, xl, xu


def _make_platypus_zcat(pid: int, n_var: int, n_obj: int):
    """Create a Platypus Problem wrapping VAMOS ZCAT evaluation."""
    from platypus import Problem, Real

    vamos_prob = _make_vamos_zcat(pid, n_var, n_obj)
    xl = np.asarray(vamos_prob.xl, dtype=float)
    xu = np.asarray(vamos_prob.xu, dtype=float)

    class _ZCATWrapper(Problem):
        def __init__(self):
            super().__init__(n_var, n_obj)
            self.types[:] = [Real(lo, hi) for lo, hi in zip(xl, xu)]

        def evaluate(self, solution):
            X = np.asarray(solution.variables, dtype=float).reshape(1, -1)
            out = {"F": np.zeros((1, n_obj), dtype=float)}
            vamos_prob.evaluate(X, out)
            solution.objectives[:] = out["F"][0]

    return _ZCATWrapper()


def _make_pygmo_zcat(pid: int, n_var: int, n_obj: int):
    """Create a PyGMO UDP wrapping VAMOS ZCAT evaluation."""
    vamos_prob = _make_vamos_zcat(pid, n_var, n_obj)
    xl = np.asarray(vamos_prob.xl, dtype=float)
    xu = np.asarray(vamos_prob.xu, dtype=float)

    class _ZCATUDP:
        def fitness(self, x):
            X = np.asarray(x, dtype=float).reshape(1, -1)
            out = {"F": np.zeros((1, n_obj), dtype=float)}
            vamos_prob.evaluate(X, out)
            return out["F"][0].tolist()

        def get_nobj(self) -> int:
            return n_obj

        def get_bounds(self):
            return xl.tolist(), xu.tolist()

        def get_name(self) -> str:
            return f"ZCAT{pid}"

    return _ZCATUDP()


def load_moead_weights(n_obj: int, pop_size: int) -> np.ndarray:
    if n_obj <= 1:
        return np.ones((pop_size, max(1, n_obj)), dtype=float)
    if n_obj == 2:
        vals = np.linspace(0.0, 1.0, pop_size, dtype=float)
        return np.column_stack([vals, 1.0 - vals])
    weight_file = MOEAD_WEIGHTS_DIR / f"W{n_obj}D_{pop_size}.dat"
    if not weight_file.exists():
        raise FileNotFoundError(f"MOEA/D weight file not found: {weight_file}")
    weights = np.loadtxt(weight_file)
    weights = np.atleast_2d(weights).astype(float, copy=False)
    return weights


def ref_front_name(pid: int, n_obj: int) -> str:
    """Reference front name for benchmark_utils."""
    if n_obj == 2:
        return f"zcat{pid}"
    return f"zcat{pid}.{n_obj}d"


def _framework_csv_name(fw: str) -> str:
    """Map framework key to the name stored in CSV."""
    if fw.startswith("vamos-"):
        return f"VAMOS ({fw.replace('vamos-', '')})"
    if fw == "jmetalpy":
        return "jMetalPy"
    if fw == "deap":
        return "DEAP"
    if fw == "platypus":
        return "Platypus"
    if fw == "pygmo":
        return "PyGMO"
    return fw


# =============================================================================
# SINGLE BENCHMARK RUN
# =============================================================================


def run_single_benchmark(
    pid: int,
    seed: int,
    framework: str,
    algorithm: str,
) -> dict[str, Any] | None:
    """Run one (problem, seed, framework, algorithm) combination."""
    problem_label = f"zcat{pid}"
    ref_name = ref_front_name(pid, N_OBJ)
    n_var = N_VAR
    n_obj = N_OBJ
    algo_display = ALGO_DISPLAY[algorithm]
    result_entry = None

    # --- VAMOS backends ---
    if framework.startswith("vamos-"):
        backend = framework.replace("vamos-", "")
        try:
            problem = _make_vamos_zcat(pid, n_var, n_obj)

            if algorithm in {"nsgaii", "nsgaii_ss", "nsgaii_archive"}:
                algo_id = "nsgaii"
                builder = (
                    NSGAIIConfig.builder()
                    .pop_size(POP_SIZE)
                    .crossover("sbx", prob=CROSSOVER_PROB, eta=CROSSOVER_ETA)
                    .mutation("polynomial", prob=1.0 / n_var, eta=MUTATION_ETA)
                    .selection("tournament")
                )
                if algorithm == "nsgaii_ss":
                    builder = builder.offspring_size(1)
                elif algorithm == "nsgaii_archive":
                    builder = builder.external_archive(capacity=None)
                algo_config = builder.build()
            elif algorithm == "smsemoa":
                algo_id = "smsemoa"
                algo_config = (
                    SMSEMOAConfig.builder()
                    .pop_size(POP_SIZE)
                    .crossover("sbx", prob=CROSSOVER_PROB, eta=CROSSOVER_ETA)
                    .mutation("polynomial", prob=1.0 / n_var, eta=MUTATION_ETA)
                    .selection("random")
                    .reference_point(offset=1.0, adaptive=True)
                    .eliminate_duplicates(True)
                    .build()
                )
            elif algorithm == "moead":
                algo_id = "moead"
                algo_config = (
                    MOEADConfig.builder()
                    .pop_size(POP_SIZE)
                    .batch_size(1)
                    .neighbor_size(MOEAD_NEIGHBOR_SIZE)
                    .delta(MOEAD_DELTA)
                    .replace_limit(MOEAD_REPLACE_LIMIT)
                    .crossover("sbx", prob=CROSSOVER_PROB, eta=CROSSOVER_ETA)
                    .mutation("polynomial", prob=1.0 / n_var, eta=MUTATION_ETA)
                    .aggregation("tchebycheff")
                    .weight_vectors(path=str(MOEAD_WEIGHTS_DIR))
                    .build()
                )
            else:
                raise ValueError(f"Unsupported algorithm '{algorithm}'.")

            if backend == "numba" and NUMBA_WARMUP_EVALS > 0:
                warmup_budget = min(int(NUMBA_WARMUP_EVALS), int(N_EVALS))
                _ = optimize(
                    problem,
                    algorithm=algo_id,
                    algorithm_config=algo_config,
                    termination=("max_evaluations", warmup_budget),
                    seed=seed,
                    engine=backend,
                )

            start = time.perf_counter()
            result = optimize(
                problem,
                algorithm=algo_id,
                algorithm_config=algo_config,
                termination=("max_evaluations", N_EVALS),
                seed=seed,
                engine=backend,
            )
            elapsed = time.perf_counter() - start

            hv_source = result.F
            n_solutions = result.X.shape[0] if result.X is not None else 0
            if algorithm == "nsgaii_archive":
                archive_payload = result.data.get("archive") if hasattr(result, "data") else None
                if isinstance(archive_payload, dict):
                    archive_F = archive_payload.get("F")
                    if archive_F is not None:
                        hv_source = _trim_for_hv(archive_F)
                        n_solutions = int(archive_F.shape[0])

            hv = compute_hv(hv_source, ref_name) if hv_source is not None else float("nan")
            igd_plus = compute_igd_plus(hv_source, ref_name) if hv_source is not None else float("nan")

            result_entry = {
                "framework": f"VAMOS ({backend})",
                "problem": problem_label,
                "algorithm": algo_display,
                "n_evals": N_EVALS,
                "seed": seed,
                "runtime_seconds": elapsed,
                "n_solutions": n_solutions,
                "hypervolume": hv,
                "igd_plus": igd_plus,
            }
            print(f"  {problem_label} VAMOS({backend}) {algorithm} seed={seed}: {elapsed:.2f}s")
        except Exception as e:
            print(f"  {problem_label} VAMOS({backend}) {algorithm} seed={seed} FAILED: {e}")

    # --- pymoo ---
    elif framework == "pymoo":
        try:
            from pymoo.algorithms.moo.nsga2 import NSGA2
            from pymoo.algorithms.moo.sms import SMSEMOA, LeastHypervolumeContributionSurvival
            from pymoo.operators.crossover.sbx import SBX
            from pymoo.operators.mutation.pm import PM
            from pymoo.optimize import minimize
            from pymoo.termination import get_termination

            pymoo_problem = _make_pymoo_zcat(pid, n_var, n_obj)
            archive_cb = None

            if algorithm in {"nsgaii", "nsgaii_ss", "nsgaii_archive"}:
                n_offsprings = 1 if algorithm == "nsgaii_ss" else None
                algo_obj = NSGA2(
                    pop_size=POP_SIZE,
                    n_offsprings=n_offsprings,
                    crossover=SBX(prob=CROSSOVER_PROB, eta=CROSSOVER_ETA),
                    mutation=PM(prob=1.0, prob_var=1.0 / n_var, eta=MUTATION_ETA),
                )
                if algorithm == "nsgaii_archive":
                    from pymoo.core.callback import Callback
                    from vamos.engine.algorithm.components.archive import UnboundedArchive

                    class ArchiveCallback(Callback):
                        def __init__(self, n_var_cb: int, n_obj_cb: int) -> None:
                            super().__init__()
                            self.archive = UnboundedArchive(n_var_cb, n_obj_cb, float)

                        def _update_archive(self, pop):
                            if pop is None:
                                return
                            X = pop.get("X")
                            F = pop.get("F")
                            if X is None or F is None:
                                return
                            self.archive.update(X, F)

                        def initialize(self, algorithm):
                            self._update_archive(getattr(algorithm, "pop", None))

                        def notify(self, algorithm):
                            self._update_archive(getattr(algorithm, "pop", None))
                            self._update_archive(getattr(algorithm, "off", None))

                    archive_cb = ArchiveCallback(n_var, n_obj)
            elif algorithm == "smsemoa":
                from pymoo.operators.selection.rnd import RandomSelection

                algo_obj = SMSEMOA(
                    pop_size=POP_SIZE,
                    n_offsprings=1,
                    selection=RandomSelection(),
                    crossover=SBX(prob=CROSSOVER_PROB, eta=CROSSOVER_ETA),
                    mutation=PM(prob=1.0, prob_var=1.0 / n_var, eta=MUTATION_ETA),
                    survival=LeastHypervolumeContributionSurvival(eps=1.0),
                    eliminate_duplicates=True,
                    normalize=False,
                )
            elif algorithm == "moead":
                from pymoo.algorithms.moo.moead import MOEAD as _PyMooMOEAD
                from pymoo.decomposition.tchebicheff import Tchebicheff
                from pymoo.core.selection import Selection

                ref_dirs = load_moead_weights(n_obj, POP_SIZE)

                class NeighborhoodSelectionWithCurrent(Selection):
                    def __init__(self, prob: float = 1.0) -> None:
                        super().__init__()
                        self.prob = float(prob)

                    def _do(self, problem, pop, n_select, n_parents, neighbors=None, random_state=None, **kwargs):
                        assert n_select == len(neighbors)
                        P = np.full((n_select, n_parents), -1, dtype=int)
                        for k in range(n_select):
                            neighbor_idx = np.asarray(neighbors[k], dtype=int)
                            if neighbor_idx.size == 0:
                                neighbor_idx = np.arange(len(pop))
                            current = int(neighbor_idx[0])
                            if random_state.random() < self.prob:
                                pool = neighbor_idx
                            else:
                                pool = np.arange(len(pop))
                            pool = pool[pool != current]
                            if pool.size < n_parents - 1:
                                pool = np.setdiff1d(np.arange(len(pop)), [current])
                            chosen = random_state.choice(pool, n_parents - 1, replace=False)
                            P[k, 0] = current
                            P[k, 1:] = chosen
                        return P

                class _MOEADNr(_PyMooMOEAD):
                    def __init__(self, *args, n_max_replace=2, **kwargs):
                        super().__init__(*args, **kwargs)
                        self._n_max_replace = n_max_replace

                    def _replace(self, k, off):
                        pop = self.pop
                        N = self.neighbors[k]
                        FV = self.decomposition.do(pop[N].get("F"), weights=self.ref_dirs[N, :], ideal_point=self.ideal)
                        off_FV = self.decomposition.do(off.F[None, :], weights=self.ref_dirs[N, :], ideal_point=self.ideal)
                        I = np.where(off_FV < FV)[0]
                        if len(I) > self._n_max_replace:
                            I = np.random.choice(I, self._n_max_replace, replace=False)
                        pop[N[I]] = off

                algo_obj = _MOEADNr(
                    ref_dirs=ref_dirs,
                    n_neighbors=MOEAD_NEIGHBOR_SIZE,
                    decomposition=Tchebicheff(),
                    prob_neighbor_mating=MOEAD_DELTA,
                    crossover=SBX(prob=CROSSOVER_PROB, eta=CROSSOVER_ETA),
                    mutation=PM(prob=1.0, prob_var=1.0 / n_var, eta=MUTATION_ETA),
                    n_max_replace=MOEAD_REPLACE_LIMIT,
                )
                algo_obj.selection = NeighborhoodSelectionWithCurrent(prob=MOEAD_DELTA)
            else:
                raise ValueError(f"Unsupported algorithm '{algorithm}' for pymoo.")

            termination = get_termination("n_eval", N_EVALS)
            minimize_kwargs: dict[str, Any] = {"seed": seed, "verbose": False}
            if archive_cb is not None:
                minimize_kwargs["callback"] = archive_cb

            start = time.perf_counter()
            res = minimize(pymoo_problem, algo_obj, termination, **minimize_kwargs)
            elapsed = time.perf_counter() - start

            hv_source = res.F
            n_solutions = res.X.shape[0] if res.X is not None else 0
            if algorithm == "nsgaii_archive" and archive_cb is not None:
                _, archive_F = archive_cb.archive.contents()
                if archive_F is not None:
                    hv_source = _trim_for_hv(archive_F)
                    n_solutions = int(archive_F.shape[0])

            hv = compute_hv(hv_source, ref_name) if hv_source is not None else float("nan")
            igd_plus = compute_igd_plus(hv_source, ref_name) if hv_source is not None else float("nan")

            result_entry = {
                "framework": "pymoo",
                "problem": problem_label,
                "algorithm": algo_display,
                "n_evals": N_EVALS,
                "seed": seed,
                "runtime_seconds": elapsed,
                "n_solutions": n_solutions,
                "hypervolume": hv,
                "igd_plus": igd_plus,
            }
            print(f"  {problem_label} pymoo {algorithm} seed={seed}: {elapsed:.2f}s")
        except Exception as e:
            print(f"  {problem_label} pymoo {algorithm} seed={seed} FAILED: {e}")

    # --- jMetalPy ---
    elif framework == "jmetalpy":
        try:
            from jmetal.algorithm.multiobjective import NSGAII
            from jmetal.algorithm.multiobjective.smsemoa import SMSEMOA as JMetalSMSEMOA
            from jmetal.algorithm.multiobjective.moead import MOEAD as JMetalMOEAD
            from jmetal.operator.crossover import SBXCrossover
            from jmetal.operator.mutation import PolynomialMutation
            from jmetal.operator.selection import NaryRandomSolutionSelection
            from jmetal.util.aggregation_function import Tschebycheff
            from jmetal.util.termination_criterion import StoppingByEvaluations

            random.seed(seed)
            np.random.seed(seed)
            try:
                from jmetal.util.random_generator import PRNG
                PRNG.seed(seed)
            except Exception:
                pass

            jmetal_prob = _make_jmetal_zcat(pid, n_var, n_obj)
            archive = None

            if algorithm in {"nsgaii", "nsgaii_ss", "nsgaii_archive"}:
                offspring_size = 1 if algorithm == "nsgaii_ss" else POP_SIZE
                if algorithm == "nsgaii_archive":
                    from vamos.engine.algorithm.components.archive import UnboundedArchive

                    class NSGAIIArchive(NSGAII):
                        def __init__(self, *args, archive_ref, **kwargs):
                            super().__init__(*args, **kwargs)
                            self._archive_ref = archive_ref

                        def evaluate(self, solutions):
                            evaluated = super().evaluate(solutions)
                            if evaluated:
                                X_vals = np.array([s.variables for s in evaluated], dtype=float)
                                F_vals = np.array([s.objectives for s in evaluated], dtype=float)
                                self._archive_ref.update(X_vals, F_vals)
                            return evaluated

                    archive = UnboundedArchive(n_var, n_obj, float)
                    jmetal_algo = NSGAIIArchive(
                        problem=jmetal_prob,
                        population_size=POP_SIZE,
                        offspring_population_size=offspring_size,
                        mutation=PolynomialMutation(probability=1.0 / n_var, distribution_index=MUTATION_ETA),
                        crossover=SBXCrossover(probability=CROSSOVER_PROB, distribution_index=CROSSOVER_ETA),
                        termination_criterion=StoppingByEvaluations(max_evaluations=N_EVALS),
                        archive_ref=archive,
                    )
                else:
                    jmetal_algo = NSGAII(
                        problem=jmetal_prob,
                        population_size=POP_SIZE,
                        offspring_population_size=offspring_size,
                        mutation=PolynomialMutation(probability=1.0 / n_var, distribution_index=MUTATION_ETA),
                        crossover=SBXCrossover(probability=CROSSOVER_PROB, distribution_index=CROSSOVER_ETA),
                        termination_criterion=StoppingByEvaluations(max_evaluations=N_EVALS),
                    )
            elif algorithm == "smsemoa":
                jmetal_algo = JMetalSMSEMOA(
                    problem=jmetal_prob,
                    population_size=POP_SIZE,
                    mutation=PolynomialMutation(probability=1.0 / n_var, distribution_index=MUTATION_ETA),
                    crossover=SBXCrossover(probability=CROSSOVER_PROB, distribution_index=CROSSOVER_ETA),
                    termination_criterion=StoppingByEvaluations(max_evaluations=N_EVALS),
                )
            elif algorithm == "moead":
                class _MOEAD_SBX(JMetalMOEAD):
                    def __init__(self, **kwargs):
                        super().__init__(**kwargs)
                        self.selection_operator = NaryRandomSolutionSelection(1)

                    def reproduction(self, mating_population):
                        offspring_population = self.crossover_operator.execute(mating_population)
                        self.mutation_operator.execute(offspring_population[0])
                        return offspring_population

                jmetal_algo = _MOEAD_SBX(
                    problem=jmetal_prob,
                    population_size=POP_SIZE,
                    mutation=PolynomialMutation(probability=1.0 / n_var, distribution_index=MUTATION_ETA),
                    crossover=SBXCrossover(probability=CROSSOVER_PROB, distribution_index=CROSSOVER_ETA),
                    aggregation_function=Tschebycheff(dimension=n_obj),
                    neighbourhood_selection_probability=MOEAD_DELTA,
                    max_number_of_replaced_solutions=MOEAD_REPLACE_LIMIT,
                    neighbor_size=MOEAD_NEIGHBOR_SIZE,
                    weight_files_path=MOEAD_WEIGHTS_DIR.as_posix(),
                    termination_criterion=StoppingByEvaluations(max_evaluations=N_EVALS),
                )
            else:
                raise ValueError(f"Unsupported algorithm '{algorithm}' for jMetalPy.")

            start = time.perf_counter()
            jmetal_algo.run()
            elapsed = time.perf_counter() - start

            if algorithm == "nsgaii_archive" and archive is not None:
                _, F = archive.contents()
                n_solutions = int(F.shape[0]) if F is not None else 0
                F_hv = _trim_for_hv(F)
                hv = compute_hv(F_hv, ref_name)
                igd_plus = compute_igd_plus(F_hv, ref_name)
            else:
                solutions = jmetal_algo.result()
                F = np.array([s.objectives for s in solutions])
                hv = compute_hv(F, ref_name)
                igd_plus = compute_igd_plus(F, ref_name)
                n_solutions = len(solutions)

            result_entry = {
                "framework": "jMetalPy",
                "problem": problem_label,
                "algorithm": algo_display,
                "n_evals": N_EVALS,
                "seed": seed,
                "runtime_seconds": elapsed,
                "n_solutions": n_solutions,
                "hypervolume": hv,
                "igd_plus": igd_plus,
            }
            print(f"  {problem_label} jMetalPy {algorithm} seed={seed}: {elapsed:.2f}s")
        except Exception as e:
            print(f"  {problem_label} jMetalPy {algorithm} seed={seed} FAILED: {e}")

    # --- DEAP ---
    elif framework == "deap":
        if algorithm not in {"nsgaii", "nsgaii_ss", "nsgaii_archive"}:
            print(f"  DEAP does not support {algorithm}, skipping")
            return None
        try:
            from deap import base, creator, tools

            problem_func, xl, xu = _make_deap_zcat(pid, n_var, n_obj)
            xl_list = xl.tolist()
            xu_list = xu.tolist()

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

            import multiprocessing as _mp
            _deap_pool = _mp.Pool(DEAP_N_JOBS) if DEAP_N_JOBS > 1 else None
            if _deap_pool is not None:
                toolbox.register("map", _deap_pool.map)

            random.seed(seed)
            pop = toolbox.population(n=POP_SIZE)
            n_gen = max(0, (N_EVALS - POP_SIZE) // POP_SIZE)
            n_offspring = max(0, N_EVALS - POP_SIZE)
            use_archive = algorithm == "nsgaii_archive"
            deap_archive = None
            if use_archive:
                from vamos.engine.algorithm.components.archive import UnboundedArchive
                deap_archive = UnboundedArchive(n_var, n_obj, float)

            start = time.perf_counter()

            invalid = [ind for ind in pop if not ind.fitness.valid]
            fitnesses = list(toolbox.map(toolbox.evaluate, invalid))
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit
            pop = toolbox.select(pop, len(pop))

            if use_archive and deap_archive is not None:
                X_pop = np.array([list(ind) for ind in pop], dtype=float)
                F_pop = np.array([ind.fitness.values for ind in pop], dtype=float)
                deap_archive.update(X_pop, F_pop)

            if algorithm == "nsgaii_ss":
                for _ in range(n_offspring):
                    parents = toolbox.select_tournament(pop, 4)
                    child1, child2 = map(toolbox.clone, parents[:2])
                    if random.random() <= CROSSOVER_PROB:
                        toolbox.mate(child1, child2)
                    child = child1
                    toolbox.mutate(child)
                    if hasattr(child, "fitness"):
                        try:
                            del child.fitness.values
                        except Exception:
                            pass
                    child.fitness.values = toolbox.evaluate(child)
                    if use_archive and deap_archive is not None:
                        deap_archive.update(
                            np.array([list(child)], dtype=float),
                            np.array([child.fitness.values], dtype=float),
                        )
                    pop = toolbox.select(pop + [child], POP_SIZE)
            else:
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
                    fitnesses = list(toolbox.map(toolbox.evaluate, invalid))
                    for ind, fit in zip(invalid, fitnesses):
                        ind.fitness.values = fit
                    if use_archive and deap_archive is not None:
                        X_off = np.array([list(ind) for ind in offspring], dtype=float)
                        F_off = np.array([ind.fitness.values for ind in offspring], dtype=float)
                        deap_archive.update(X_off, F_off)
                    pop = toolbox.select(pop + offspring, POP_SIZE)

            elapsed = time.perf_counter() - start

            if _deap_pool is not None:
                _deap_pool.close()
                _deap_pool.join()

            if use_archive and deap_archive is not None:
                _, F = deap_archive.contents()
                n_solutions = int(F.shape[0]) if F is not None else 0
                F_hv = _trim_for_hv(F)
                hv = compute_hv(F_hv, ref_name)
                igd_plus = compute_igd_plus(F_hv, ref_name)
            else:
                fronts = tools.sortNondominated(pop, len(pop), first_front_only=True)
                F = np.array([ind.fitness.values for ind in fronts[0]])
                hv = compute_hv(F, ref_name)
                igd_plus = compute_igd_plus(F, ref_name)
                n_solutions = len(fronts[0])

            result_entry = {
                "framework": "DEAP",
                "problem": problem_label,
                "algorithm": algo_display,
                "n_evals": N_EVALS,
                "seed": seed,
                "runtime_seconds": elapsed,
                "n_solutions": n_solutions,
                "hypervolume": hv,
                "igd_plus": igd_plus,
            }
            print(f"  {problem_label} DEAP {algorithm} seed={seed}: {elapsed:.2f}s")
        except Exception as e:
            print(f"  {problem_label} DEAP {algorithm} seed={seed} FAILED: {e}")

    # --- Platypus ---
    elif framework == "platypus":
        if algorithm not in {"nsgaii", "nsgaii_archive", "moead"}:
            print(f"  Platypus does not support {algorithm}, skipping")
            return None
        try:
            from platypus import MOEAD as PlatypusMOEAD
            from platypus import NSGAII as PlatypusNSGAII, Archive, GAOperator
            from platypus import PM as PlatypusPM, SBX as PlatypusSBX, TournamentSelector

            random.seed(seed)
            platypus_problem = _make_platypus_zcat(pid, n_var, n_obj)

            if algorithm in {"nsgaii", "nsgaii_archive"}:
                plat_archive = Archive() if algorithm == "nsgaii_archive" else None
                variator = GAOperator(
                    PlatypusSBX(probability=CROSSOVER_PROB, distribution_index=CROSSOVER_ETA),
                    PlatypusPM(probability=1.0 / n_var, distribution_index=MUTATION_ETA),
                )
                plat_algo = PlatypusNSGAII(
                    platypus_problem,
                    population_size=POP_SIZE,
                    selector=TournamentSelector(2),
                    variator=variator,
                    archive=plat_archive,
                )
            elif algorithm == "moead":
                def _weight_generator(nobjs, population_size=POP_SIZE):
                    weights = load_moead_weights(nobjs, population_size)
                    return weights.tolist()

                variator = GAOperator(
                    PlatypusSBX(probability=CROSSOVER_PROB, distribution_index=CROSSOVER_ETA),
                    PlatypusPM(probability=1.0 / n_var, distribution_index=MUTATION_ETA),
                )
                plat_algo = PlatypusMOEAD(
                    platypus_problem,
                    neighborhood_size=MOEAD_NEIGHBOR_SIZE,
                    delta=MOEAD_DELTA,
                    eta=MOEAD_REPLACE_LIMIT,
                    variator=variator,
                    weight_generator=_weight_generator,
                    population_size=POP_SIZE,
                )
            else:
                raise ValueError(f"Unsupported algorithm '{algorithm}' for Platypus.")

            start = time.perf_counter()
            plat_algo.run(N_EVALS)
            elapsed = time.perf_counter() - start

            result_solutions = list(plat_algo.result)
            F = np.array([s.objectives for s in result_solutions])
            F_hv = _trim_for_hv(F) if algorithm == "nsgaii_archive" else F
            hv = compute_hv(F_hv, ref_name)
            igd_plus = compute_igd_plus(F_hv, ref_name)

            result_entry = {
                "framework": "Platypus",
                "problem": problem_label,
                "algorithm": algo_display,
                "n_evals": N_EVALS,
                "seed": seed,
                "runtime_seconds": elapsed,
                "n_solutions": len(result_solutions),
                "hypervolume": hv,
                "igd_plus": igd_plus,
            }
            print(f"  {problem_label} Platypus {algorithm} seed={seed}: {elapsed:.2f}s")
        except Exception as e:
            print(f"  {problem_label} Platypus {algorithm} seed={seed} FAILED: {e}")

    # --- PyGMO ---
    elif framework == "pygmo":
        if algorithm != "nsgaii":
            print(f"  PyGMO does not support {algorithm}, skipping")
            return None
        try:
            import pygmo as pg

            random.seed(seed)
            np.random.seed(seed)

            pygmo_problem = pg.problem(_make_pygmo_zcat(pid, n_var, n_obj))
            generations = max(0, N_EVALS // POP_SIZE - 1)

            # pygmo requires cr strictly < 1.0
            cr = min(CROSSOVER_PROB, 1.0 - 1e-16)
            uda = pg.nsga2(
                gen=generations,
                seed=seed,
                cr=cr,
                eta_c=CROSSOVER_ETA,
                m=1.0 / n_var,
                eta_m=MUTATION_ETA,
            )
            algo = pg.algorithm(uda)
            pop = pg.population(pygmo_problem, size=POP_SIZE, seed=seed)

            start = time.perf_counter()
            pop = algo.evolve(pop)
            elapsed = time.perf_counter() - start

            F = np.asarray(pop.get_f(), dtype=float)
            hv = compute_hv(F, ref_name)
            igd_plus = compute_igd_plus(F, ref_name)

            result_entry = {
                "framework": "PyGMO",
                "problem": problem_label,
                "algorithm": algo_display,
                "n_evals": N_EVALS,
                "seed": seed,
                "runtime_seconds": elapsed,
                "n_solutions": int(F.shape[0]),
                "hypervolume": hv,
                "igd_plus": igd_plus,
            }
            print(f"  {problem_label} PyGMO {algorithm} seed={seed}: {elapsed:.2f}s")
        except Exception as e:
            print(f"  {problem_label} PyGMO {algorithm} seed={seed} FAILED: {e}")

    else:
        print(f"  Unknown framework: {framework}")

    return result_entry


# =============================================================================
# SAVE HELPERS
# =============================================================================


def _save_partial(results_list: list[dict[str, Any]], output_csv: Path) -> None:
    """Persist results to CSV, merging with existing data."""
    filtered = [r for r in results_list if r is not None]
    if not filtered:
        return
    df_new = pd.DataFrame(filtered)
    if output_csv.exists() and RESUME:
        try:
            df_old = pd.read_csv(output_csv)
        except Exception as exc:
            print(f"Warning: failed to read existing CSV ({output_csv}): {exc}")
            df = df_new
        else:
            df = pd.concat([df_old, df_new], ignore_index=True)
            key_cols = [c for c in ["framework", "problem", "algorithm", "n_evals", "seed"] if c in df.columns]
            if key_cols:
                df = df.drop_duplicates(subset=key_cols, keep="last")
    else:
        df = df_new
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)


def _dedupe_existing_csv(output_csv: Path) -> int:
    """Deduplicate an existing CSV in-place by benchmark key columns."""
    if not output_csv.is_file():
        return 0
    try:
        df = pd.read_csv(output_csv)
    except Exception as exc:
        print(f"Warning: could not read {output_csv} for dedupe: {exc}")
        return 0

    required = ["framework", "problem", "algorithm", "n_evals", "seed"]
    key_cols = [c for c in required if c in df.columns]
    if not key_cols:
        return 0

    before_raw = len(df)

    # Drop accidental header-as-row artifacts and invalid numeric keys.
    if set(required).issubset(df.columns):
        fw = df["framework"].astype(str).str.strip().str.lower()
        pr = df["problem"].astype(str).str.strip().str.lower()
        alg = df["algorithm"].astype(str).str.strip().str.lower()
        bad_header = (
            fw.eq("framework")
            | pr.eq("problem")
            | alg.eq("algorithm")
        )
        if bad_header.any():
            df = df.loc[~bad_header].copy()

        df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
        df["n_evals"] = pd.to_numeric(df["n_evals"], errors="coerce")
        df = df.dropna(subset=["seed", "n_evals"])
        df["seed"] = df["seed"].astype(int)
        df["n_evals"] = df["n_evals"].astype(int)

    df = df.drop_duplicates(subset=key_cols, keep="last")
    removed = before_raw - len(df)
    if removed > 0:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
    return removed


def _load_completed_keys(output_csv: Path, algo_display: str) -> set[tuple[str, str, int]]:
    """Load already-completed (framework, problem, seed) combos."""
    keys: set[tuple[str, str, int]] = set()
    if not (RESUME and output_csv.is_file()):
        return keys
    try:
        df = pd.read_csv(output_csv)
        required_cols = {"framework", "problem", "algorithm", "n_evals", "seed"}
        if not required_cols.issubset(df.columns):
            return keys
        # Filter to matching algorithm and budget
        df = df[(df["algorithm"] == algo_display) & (df["n_evals"] == N_EVALS)]
        # Only consider ZCAT problems
        df = df[df["problem"].astype(str).str.startswith("zcat")]
        for _, row in df.iterrows():
            try:
                keys.add((str(row["framework"]), str(row["problem"]), int(row["seed"])))
            except Exception:
                continue
    except Exception as e:
        print(f"Warning: could not load {output_csv} for resume: {e}")
    return keys


def _run_chunk_threadpool(chunk: list[dict[str, Any]], n_jobs: int) -> list[dict[str, Any] | None]:
    """Run one chunk with native threads (no multiprocessing backend)."""
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(run_single_benchmark, **task) for task in chunk]
        return [future.result() for future in futures]


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    print("ZCAT Benchmark for All Tables")
    print(f"  Problems:    ZCAT {PROBLEM_IDS}")
    print(f"  Objectives:  {N_OBJ}")
    print(f"  Variables:   {N_VAR}")
    print(f"  Algorithms:  {ALGORITHMS}")
    print(f"  n_evals:     {N_EVALS:,}")
    print(f"  seeds:       {N_SEEDS}")
    print(f"  workers:     {N_JOBS}")
    print(f"  save every:  {SAVE_EVERY}")

    for algorithm in ALGORITHMS:
        if algorithm not in ALGO_CSV_MAP:
            print(f"  [SKIP] Unknown algorithm: {algorithm}")
            continue

        output_csv = ALGO_CSV_MAP[algorithm]
        algo_display = ALGO_DISPLAY[algorithm]

        # Determine frameworks for this algorithm
        if _fw_override:
            frameworks = _unique_preserve_order(_parse_str_list(_fw_override))
        else:
            frameworks = _unique_preserve_order(ALGO_FRAMEWORKS.get(algorithm, ["vamos-numba", "pymoo", "jmetalpy"]))

        print(f"\n{'='*60}")
        print(f"Algorithm: {algo_display} ({algorithm})")
        print(f"Output:    {output_csv}")
        print(f"Frameworks: {frameworks}")
        print(f"{'='*60}")

        if RESUME:
            removed = _dedupe_existing_csv(output_csv)
            if removed > 0:
                print(f"Resume cleanup: removed {removed} duplicate rows from {output_csv.name}")

        # Load completed keys for resume
        completed = _load_completed_keys(output_csv, algo_display)
        if completed:
            print(f"Resume: {len(completed)} ZCAT runs already completed")

        # Build task list
        tasks: list[dict[str, Any]] = []
        for pid in PROBLEM_IDS:
            for seed in range(N_SEEDS):
                for fw in frameworks:
                    fw_csv = _framework_csv_name(fw)
                    if (fw_csv, f"zcat{pid}", seed) in completed:
                        continue
                    tasks.append({"pid": pid, "seed": seed, "framework": fw, "algorithm": algorithm})

        print(f"Total runs: {len(tasks)}")
        if not tasks:
            print("Nothing to do for this algorithm.")
            continue

        # Run in chunks with periodic saves
        results_list: list[dict[str, Any]] = []
        use_thread_backend = False
        for i in range(0, len(tasks), SAVE_EVERY):
            chunk = tasks[i : i + SAVE_EVERY]
            with joblib_progress(total=len(chunk), desc=f"ZCAT {algorithm} [{i}..{i+len(chunk)}/{len(tasks)}]"):
                if use_thread_backend:
                    chunk_results = _run_chunk_threadpool(chunk, N_JOBS)
                else:
                    try:
                        chunk_results = Parallel(n_jobs=N_JOBS, batch_size=1)(
                            delayed(run_single_benchmark)(**t) for t in chunk
                        )
                    except PermissionError as exc:
                        if N_JOBS <= 1:
                            raise
                        use_thread_backend = True
                        print(
                            "Warning: multiprocessing backend unavailable "
                            f"({exc}); retrying with ThreadPoolExecutor backend."
                        )
                        chunk_results = _run_chunk_threadpool(chunk, N_JOBS)
            results_list.extend([r for r in chunk_results if r is not None])
            _save_partial(results_list, output_csv)

        n_ok = len(results_list)
        print(f"Completed {n_ok} runs for {algo_display}, saved to {output_csv}")

    print("\nDone!")


if __name__ == "__main__":
    main()
