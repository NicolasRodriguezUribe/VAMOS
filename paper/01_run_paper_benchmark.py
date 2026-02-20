"""
VAMOS Paper Benchmark Script
===========================
Runs complete benchmark and writes CSV results for the paper.

Usage: python paper/01_run_paper_benchmark.py

Use 04_update_paper_tables_from_csv.py to generate and inject LaTeX tables.

Environment variables:
  - VAMOS_PAPER_ALGORITHM: nsgaii, nsgaii-ss, nsgaii-archive, smsemoa, moead,
    or a comma list (e.g., nsgaii-ss,smsemoa). If unset, runs nsgaii-ss + nsgaii-archive.
  - VAMOS_PAPER_ALGORITHMS: comma-separated list to run sequentially
    (e.g., nsgaii,nsgaii-ss,nsgaii-archive,smsemoa,moead)
  - VAMOS_N_EVALS: evaluations per run (default: 50000)
  - VAMOS_N_SEEDS: number of seeds (default: 30)
  - VAMOS_N_JOBS: joblib workers (default: CPU count - 1)
  - VAMOS_PAPER_FRAMEWORKS: comma-separated framework keys (default: all)
  - VAMOS_PAPER_OUTPUT_CSV: output CSV path (default: experiments/benchmark_paper*.csv)
  - VAMOS_NUMBA_WARMUP_EVALS: warmup evaluations before timing Numba runs (default: 2000)
  - VAMOS_OBJECTIVE_ALIGNMENT_CHECK: 1/0 run objective-alignment preflight (default: 1)
  - VAMOS_OBJECTIVE_ALIGNMENT_SAMPLES: random points per problem (default: 64)
  - VAMOS_OBJECTIVE_ALIGNMENT_RTOL: relative tolerance (default: 1e-6)
  - VAMOS_OBJECTIVE_ALIGNMENT_ATOL: absolute tolerance (default: 1e-8)
"""

import os
import sys
from pathlib import Path
import subprocess

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

from vamos.foundation.data import weight_path


def _normalize_algorithm(value: str) -> str:
    algo = value.strip().lower()
    if algo in {"nsgaii", "nsga2", "nsga-ii", "nsga_ii"}:
        return "nsgaii"
    if algo in {
        "nsgaii_ss",
        "nsgaii-ss",
        "nsga2-ss",
        "nsga2_ss",
        "nsgaii_steady",
        "nsgaii-steady",
        "nsgaii_steady_state",
        "nsgaii-steady-state",
    }:
        return "nsgaii_ss"
    if algo in {
        "nsgaii_archive",
        "nsgaii-archive",
        "nsga2-archive",
        "nsga2_archive",
        "nsgaii_external_archive",
        "nsgaii-external-archive",
        "nsgaii_unbounded_archive",
        "nsgaii-unbounded-archive",
    }:
        return "nsgaii_archive"
    if algo in {"smsemoa", "sms-emoa", "sms_emoa"}:
        return "smsemoa"
    if algo in {"moead", "moea/d", "moea-d", "moea_d"}:
        return "moead"
    raise ValueError(
        f"Unsupported algorithm '{value}'. Expected nsgaii, nsgaii-ss, nsgaii-archive, smsemoa, or moead."
    )


_algo_list_env = os.environ.get("VAMOS_PAPER_ALGORITHMS")
if not _algo_list_env:
    _algo_single_raw = os.environ.get("VAMOS_PAPER_ALGORITHM")
    _algo_single = (_algo_single_raw or "").strip().lower()
    if not _algo_single:
        _algo_list_env = "nsgaii_ss,nsgaii_archive"
    elif "," in _algo_single:
        _algo_list_env = _algo_single
    elif _algo_single in {"sms-moead", "smsemoa+moead", "smsemoa_moead", "moead+smsemoa", "moead-smsemoa"}:
        _algo_list_env = "smsemoa,moead"
    elif _algo_single in {"all", "full", "paper"}:
        _algo_list_env = "nsgaii_ss,nsgaii_archive"

if _algo_list_env:
    raw_list = [item.strip() for item in _algo_list_env.split(",") if item.strip()]
    if not raw_list:
        raise ValueError("VAMOS_PAPER_ALGORITHMS is set but empty.")
    normalized = []
    seen = set()
    for item in raw_list:
        algo = _normalize_algorithm(item)
        if algo not in seen:
            normalized.append(algo)
            seen.add(algo)

    print(f"Launching sequential benchmarks: {normalized}")
    for algo in normalized:
        env = os.environ.copy()
        env["VAMOS_PAPER_ALGORITHM"] = algo
        env.pop("VAMOS_PAPER_ALGORITHMS", None)
        print("\n" + "=" * 60)
        print(f"RUNNING {algo.upper()}")
        print("=" * 60)
        subprocess.run([sys.executable, __file__], env=env, check=True)
    raise SystemExit(0)

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "experiments"

# Algorithm to benchmark: "nsgaii", "nsgaii_ss", "nsgaii_archive", "smsemoa", or "moead".
_algo_env = os.environ.get("VAMOS_PAPER_ALGORITHM", "nsgaii")
ALGORITHM = _normalize_algorithm(_algo_env)

ALGORITHM_DISPLAY = {
    "nsgaii": "NSGA-II",
    "nsgaii_ss": "NSGA-II (steady-state)",
    "nsgaii_archive": "NSGA-II (ext. archive)",
    "smsemoa": "SMS-EMOA",
    "moead": "MOEA/D",
}[ALGORITHM]

# Problems to benchmark (by family)
ZDT_PROBLEMS = ["zdt1", "zdt2", "zdt3", "zdt4", "zdt6"]
DTLZ_PROBLEMS = ["dtlz1", "dtlz2", "dtlz3", "dtlz4", "dtlz5", "dtlz6", "dtlz7"]
WFG_PROBLEMS = ["wfg1", "wfg2", "wfg3", "wfg4", "wfg5", "wfg6", "wfg7", "wfg8", "wfg9"]

USE_ZDT = True
USE_DTLZ = True
USE_WFG = True

N_EVALS = int(os.environ.get("VAMOS_N_EVALS", "50000"))
N_SEEDS = int(os.environ.get("VAMOS_N_SEEDS", "30"))
if ALGORITHM == "nsgaii":
    _default_output = DATA_DIR / "benchmark_paper.csv"
elif ALGORITHM == "nsgaii_ss":
    _default_output = DATA_DIR / "benchmark_paper_nsgaii_ss.csv"
elif ALGORITHM == "nsgaii_archive":
    _default_output = DATA_DIR / "benchmark_paper_nsgaii_archive.csv"
elif ALGORITHM == "smsemoa":
    _default_output = DATA_DIR / "benchmark_paper_smsemoa.csv"
else:
    _default_output = DATA_DIR / "benchmark_paper_moead.csv"
OUTPUT_CSV = Path(os.environ.get("VAMOS_PAPER_OUTPUT_CSV", str(_default_output)))
NUMBA_WARMUP_EVALS = int(os.environ.get("VAMOS_NUMBA_WARMUP_EVALS", "2000"))
ALIGN_CHECK_ENABLED = bool(int(os.environ.get("VAMOS_OBJECTIVE_ALIGNMENT_CHECK", "1")))
ALIGN_CHECK_SAMPLES = int(os.environ.get("VAMOS_OBJECTIVE_ALIGNMENT_SAMPLES", "64"))
ALIGN_CHECK_RTOL = float(os.environ.get("VAMOS_OBJECTIVE_ALIGNMENT_RTOL", "1e-6"))
ALIGN_CHECK_ATOL = float(os.environ.get("VAMOS_OBJECTIVE_ALIGNMENT_ATOL", "1e-8"))

POP_SIZE = 100
CROSSOVER_PROB = 1.0
CROSSOVER_ETA = 20.0
MUTATION_ETA = 20.0
MOEAD_NEIGHBOR_SIZE = 20
MOEAD_DELTA = 0.9
MOEAD_REPLACE_LIMIT = MOEAD_NEIGHBOR_SIZE
MOEAD_DE_CR = 1.0
MOEAD_DE_F = 0.5
MOEAD_PBI_THETA = 5.0

ZDT_N_VAR = {"zdt1": 30, "zdt2": 30, "zdt3": 30, "zdt4": 10, "zdt6": 10}
DTLZ_N_VAR = {"dtlz1": 7, "dtlz2": 12, "dtlz3": 12, "dtlz4": 12, "dtlz5": 12, "dtlz6": 12, "dtlz7": 22}
WFG_N_VAR = 24
ZDT_N_OBJ = 2
DTLZ_N_OBJ = 3
WFG_N_OBJ = 2
MOEAD_WEIGHTS_DIR = weight_path("W3D_100.dat").parent

# Frameworks to benchmark
DEFAULT_FRAMEWORKS_NS = [
    "vamos-numba",
    "pymoo",
    "jmetalpy",
]
DEFAULT_FRAMEWORKS_NS_SS = [
    "vamos-numba",
    "pymoo",
    "jmetalpy",
]
DEFAULT_FRAMEWORKS_NS_ARCHIVE = [
    "vamos-numba",
    "pymoo",
    "jmetalpy",
]
DEFAULT_FRAMEWORKS_SMS = [
    "vamos-numba",
    "pymoo",
    "jmetalpy",
]
DEFAULT_FRAMEWORKS_MOEAD = [
    "vamos-numba",
    "pymoo",
    "jmetalpy",
]
if ALGORITHM == "nsgaii":
    DEFAULT_FRAMEWORKS = DEFAULT_FRAMEWORKS_NS
elif ALGORITHM == "nsgaii_ss":
    DEFAULT_FRAMEWORKS = DEFAULT_FRAMEWORKS_NS_SS
elif ALGORITHM == "nsgaii_archive":
    DEFAULT_FRAMEWORKS = DEFAULT_FRAMEWORKS_NS_ARCHIVE
elif ALGORITHM == "smsemoa":
    DEFAULT_FRAMEWORKS = DEFAULT_FRAMEWORKS_SMS
else:
    DEFAULT_FRAMEWORKS = DEFAULT_FRAMEWORKS_MOEAD
_frameworks_env = os.environ.get("VAMOS_PAPER_FRAMEWORKS")
FRAMEWORKS = [f.strip() for f in _frameworks_env.split(",") if f.strip()] if _frameworks_env else DEFAULT_FRAMEWORKS

# Platypus is not supported for the steady-state NSGA-II baseline.
# Ensure it's not present when running the steady-state experiment.
if ALGORITHM == "nsgaii_ss":
    FRAMEWORKS = [f for f in FRAMEWORKS if f != "platypus"]

# Build problem list
PROBLEMS = []
if USE_ZDT:
    PROBLEMS.extend(ZDT_PROBLEMS)
if USE_DTLZ:
    PROBLEMS.extend(DTLZ_PROBLEMS)
if USE_WFG:
    PROBLEMS.extend(WFG_PROBLEMS)

_problems_env = os.environ.get("VAMOS_PAPER_PROBLEMS")
if _problems_env:
    requested = [p.strip().lower() for p in _problems_env.split(",") if p.strip()]
    if not requested:
        raise ValueError("VAMOS_PAPER_PROBLEMS is set but empty.")
    requested_set = set(requested)
    unknown = [p for p in requested if p not in set(p.lower() for p in PROBLEMS)]
    if unknown:
        raise ValueError(f"Unknown problems in VAMOS_PAPER_PROBLEMS: {unknown}")
    PROBLEMS = [p for p in PROBLEMS if p.lower() in requested_set]

print(f"Configured {len(PROBLEMS)} problems: {PROBLEMS}")
print(f"Algorithm: {ALGORITHM_DISPLAY} ({ALGORITHM})")
print(f"Frameworks: {FRAMEWORKS}")
print(f"Evaluations per run: {N_EVALS:,}")
print(f"Seeds: {N_SEEDS}")
print(f"Total runs: {len(PROBLEMS) * len(FRAMEWORKS) * N_SEEDS}")

# =============================================================================
# PARALLEL BENCHMARK EXECUTION
# =============================================================================

from joblib import Parallel, delayed

# Use all cores minus 1
# joblib supports negative n_jobs (e.g., -1 = all cores). Only n_jobs==1 is truly sequential.
_default_n_jobs = str(max(1, (os.cpu_count() or 2) - 1))
N_JOBS = int(os.environ.get("VAMOS_N_JOBS", _default_n_jobs))
if N_JOBS == 0:
    raise ValueError("VAMOS_N_JOBS cannot be 0 (joblib expects 1, -1, or another non-zero integer).")
print(f"Using {N_JOBS} parallel workers")
DEAP_N_JOBS = int(os.environ.get("VAMOS_DEAP_N_JOBS", "2"))
SAVE_EVERY = int(os.environ.get("VAMOS_PAPER_SAVE_EVERY", "50"))
RESUME = os.environ.get("VAMOS_PAPER_RESUME", "1").strip().lower() not in {"0", "false", "no"}

from vamos.foundation.problem.registry import make_problem_selection
from vamos import optimize
from vamos.engine.algorithm.config import MOEADConfig, NSGAIIConfig, SMSEMOAConfig

try:
    from .benchmark_utils import compute_hv, compute_igd_plus
except ImportError:
    from benchmark_utils import compute_hv, compute_igd_plus

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
    if weights.shape[0] != pop_size or weights.shape[1] != n_obj:
        raise ValueError(f"Expected weights shape ({pop_size}, {n_obj}) in {weight_file}, got {weights.shape}.")
    return weights


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
            if ALGORITHM in {"nsgaii", "nsgaii_ss", "nsgaii_archive"}:
                algo_id = "nsgaii"
                builder = (
                    NSGAIIConfig.builder()
                    .pop_size(POP_SIZE)
                    .crossover("sbx", prob=CROSSOVER_PROB, eta=CROSSOVER_ETA)
                    .mutation("pm", prob=1.0 / n_var, eta=MUTATION_ETA)
                    .selection("tournament")
                )
                if ALGORITHM == "nsgaii_ss":
                    builder = builder.steady_state(True).offspring_size(1).replacement_size(1)
                elif ALGORITHM == "nsgaii_archive":
                    builder = builder.external_archive(capacity=None)
                algo_config = builder.build()
            elif ALGORITHM == "smsemoa":
                algo_id = "smsemoa"
                algo_config = (
                    SMSEMOAConfig.builder()
                    .pop_size(POP_SIZE)
                    .crossover("sbx", prob=CROSSOVER_PROB, eta=CROSSOVER_ETA)
                    .mutation("pm", prob=1.0 / n_var, eta=MUTATION_ETA)
                    .selection("random")
                    .reference_point(offset=1.0, adaptive=True)
                    .eliminate_duplicates(True)
                    .build()
                )
            elif ALGORITHM == "moead":
                algo_id = "moead"
                algo_config = (
                    MOEADConfig.builder()
                    .pop_size(POP_SIZE)
                    .batch_size(1)
                    .neighbor_size(MOEAD_NEIGHBOR_SIZE)
                    .delta(MOEAD_DELTA)
                    .replace_limit(MOEAD_REPLACE_LIMIT)
                    .crossover("de", cr=MOEAD_DE_CR, f=MOEAD_DE_F)
                    .mutation("pm", prob=1.0 / n_var, eta=MUTATION_ETA)
                    .aggregation("pbi", theta=MOEAD_PBI_THETA)
                    .weight_vectors(path=str(MOEAD_WEIGHTS_DIR))
                    .build()
                )
            else:
                raise ValueError(f"Unsupported algorithm '{ALGORITHM}'.")

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
            if ALGORITHM == "nsgaii_archive":
                archive_payload = result.data.get("archive") if hasattr(result, "data") else None
                if isinstance(archive_payload, dict):
                    archive_F = archive_payload.get("F")
                    if archive_F is not None:
                        hv_source = archive_F
                        n_solutions = int(archive_F.shape[0])
            hv = compute_hv(hv_source, problem_name) if hv_source is not None else float("nan")
            igd_plus = compute_igd_plus(hv_source, problem_name) if hv_source is not None else float("nan")
            result_entry = {
                "framework": f"VAMOS ({backend})",
                "problem": problem_name,
                "algorithm": ALGORITHM_DISPLAY,
                "n_evals": N_EVALS,
                "seed": seed,
                "runtime_seconds": elapsed,
                "n_solutions": n_solutions,
                "hypervolume": hv,
                "igd_plus": igd_plus,
            }
            print(f"  {problem_name} VAMOS({backend}) seed={seed}: {elapsed:.2f}s")
        except Exception as e:
            print(f"  {problem_name} VAMOS({backend}) seed={seed} FAILED: {e}")

    # pymoo
    elif framework == "pymoo":
        try:
            from pymoo.algorithms.moo.nsga2 import NSGA2
            from pymoo.algorithms.moo.sms import SMSEMOA, LeastHypervolumeContributionSurvival
            from pymoo.optimize import minimize
            from pymoo.termination import get_termination
            from pymoo.problems import get_problem
            from pymoo.operators.crossover.sbx import SBX
            from pymoo.operators.mutation.pm import PM
            from pymoo.operators.selection.rnd import RandomSelection

            if problem_name.startswith("zdt"):
                pymoo_problem = get_problem(problem_name, n_var=n_var)
            else:
                pymoo_problem = get_problem(problem_name, n_var=n_var, n_obj=n_obj)

            archive_cb = None
            if ALGORITHM in {"nsgaii", "nsgaii_ss", "nsgaii_archive"}:
                n_offsprings = 1 if ALGORITHM == "nsgaii_ss" else None
                algorithm = NSGA2(
                    pop_size=POP_SIZE,
                    n_offsprings=n_offsprings,
                    crossover=SBX(prob=CROSSOVER_PROB, eta=CROSSOVER_ETA),
                    mutation=PM(prob=1.0, prob_var=1.0 / n_var, eta=MUTATION_ETA),
                )
                if ALGORITHM == "nsgaii_archive":
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

                        def initialize(self, algorithm):  # type: ignore[override]
                            self._update_archive(getattr(algorithm, "pop", None))

                        def notify(self, algorithm):  # type: ignore[override]
                            self._update_archive(getattr(algorithm, "pop", None))
                            self._update_archive(getattr(algorithm, "off", None))

                    archive_cb = ArchiveCallback(n_var, n_obj)
            elif ALGORITHM == "smsemoa":
                algorithm = SMSEMOA(
                    pop_size=POP_SIZE,
                    n_offsprings=1,
                    selection=RandomSelection(),
                    crossover=SBX(prob=CROSSOVER_PROB, eta=CROSSOVER_ETA),
                    mutation=PM(prob=1.0, prob_var=1.0 / n_var, eta=MUTATION_ETA),
                    survival=LeastHypervolumeContributionSurvival(eps=1.0),
                    eliminate_duplicates=True,
                    normalize=False,
                )
            else:
                from pymoo.algorithms.moo.moead import MOEAD
                from pymoo.operators.crossover.dex import DEX
                from pymoo.decomposition.pbi import PBI
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

                algorithm = MOEAD(
                    ref_dirs=ref_dirs,
                    n_neighbors=MOEAD_NEIGHBOR_SIZE,
                    decomposition=PBI(theta=MOEAD_PBI_THETA),
                    prob_neighbor_mating=MOEAD_DELTA,
                    crossover=DEX(CR=MOEAD_DE_CR, F=MOEAD_DE_F, variant="bin"),
                    mutation=PM(prob=1.0, prob_var=1.0 / n_var, eta=MUTATION_ETA),
                )
                algorithm.selection = NeighborhoodSelectionWithCurrent(prob=MOEAD_DELTA)
            termination = get_termination("n_eval", N_EVALS)

            start = time.perf_counter()
            minimize_kwargs = {"seed": seed, "verbose": False}
            if archive_cb is not None:
                minimize_kwargs["callback"] = archive_cb
            res = minimize(
                pymoo_problem,
                algorithm,
                termination,
                **minimize_kwargs,
            )
            elapsed = time.perf_counter() - start
            hv_source = res.F
            n_solutions = res.X.shape[0] if res.X is not None else 0
            if ALGORITHM == "nsgaii_archive" and archive_cb is not None:
                _, archive_F = archive_cb.archive.contents()
                if archive_F is not None:
                    hv_source = archive_F
                    n_solutions = int(archive_F.shape[0])
            hv = compute_hv(hv_source, problem_name) if hv_source is not None else float("nan")
            igd_plus = compute_igd_plus(hv_source, problem_name) if hv_source is not None else float("nan")
            result_entry = {
                "framework": "pymoo",
                "problem": problem_name,
                "algorithm": ALGORITHM_DISPLAY,
                "n_evals": N_EVALS,
                "seed": seed,
                "runtime_seconds": elapsed,
                "n_solutions": n_solutions,
                "hypervolume": hv,
                "igd_plus": igd_plus,
            }
            print(f"  {problem_name} pymoo seed={seed}: {elapsed:.2f}s")
        except Exception as e:
            print(f"  {problem_name} pymoo seed={seed} FAILED: {e}")

    # DEAP
    elif framework == "deap":
        if ALGORITHM not in {"nsgaii", "nsgaii_ss", "nsgaii_archive"}:
            raise ValueError("DEAP baseline is only implemented for NSGA-II variants.")
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

            import multiprocessing as _mp
            _deap_pool = _mp.Pool(DEAP_N_JOBS) if DEAP_N_JOBS > 1 else None
            if _deap_pool is not None:
                toolbox.register("map", _deap_pool.map)

            random.seed(seed)
            pop = toolbox.population(n=POP_SIZE)
            n_gen = max(0, (N_EVALS - POP_SIZE) // POP_SIZE)
            n_offspring = max(0, N_EVALS - POP_SIZE)
            use_archive = ALGORITHM == "nsgaii_archive"
            archive = None
            if use_archive:
                from vamos.engine.algorithm.components.archive import UnboundedArchive

                archive = UnboundedArchive(n_var, n_obj, float)

            start = time.perf_counter()

            # Evaluate the initial population
            invalid = [ind for ind in pop if not ind.fitness.valid]
            fitnesses = list(toolbox.map(toolbox.evaluate, invalid))
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit

            # Assign crowding distance
            pop = toolbox.select(pop, len(pop))

            if use_archive and archive is not None:
                X_pop = np.array([list(ind) for ind in pop], dtype=float)
                F_pop = np.array([ind.fitness.values for ind in pop], dtype=float)
                archive.update(X_pop, F_pop)

            if ALGORITHM == "nsgaii_ss":
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
                    if use_archive and archive is not None:
                        archive.update(np.array([list(child)], dtype=float), np.array([child.fitness.values], dtype=float))
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

                    if use_archive and archive is not None:
                        X_off = np.array([list(ind) for ind in offspring], dtype=float)
                        F_off = np.array([ind.fitness.values for ind in offspring], dtype=float)
                        archive.update(X_off, F_off)

                    pop = toolbox.select(pop + offspring, POP_SIZE)

            elapsed = time.perf_counter() - start

            if _deap_pool is not None:
                _deap_pool.close()
                _deap_pool.join()

            if use_archive and archive is not None:
                _, F = archive.contents()
                hv = compute_hv(F, problem_name)
                igd_plus = compute_igd_plus(F, problem_name)
                n_solutions = int(F.shape[0]) if F is not None else 0
            else:
                fronts = tools.sortNondominated(pop, len(pop), first_front_only=True)
                F = np.array([ind.fitness.values for ind in fronts[0]])
                hv = compute_hv(F, problem_name)
                igd_plus = compute_igd_plus(F, problem_name)
                n_solutions = len(fronts[0])

            result_entry = {
                "framework": "DEAP",
                "problem": problem_name,
                "algorithm": ALGORITHM_DISPLAY,
                "n_evals": N_EVALS,
                "seed": seed,
                "runtime_seconds": elapsed,
                "n_solutions": n_solutions,
                "hypervolume": hv,
                "igd_plus": igd_plus,
            }
            print(f"  {problem_name} DEAP seed={seed}: {elapsed:.2f}s")
        except Exception as e:
            print(f"  {problem_name} DEAP seed={seed} FAILED: {e}")

    # jMetalPy
    elif framework == "jmetalpy":
        try:
            from jmetal.algorithm.multiobjective import NSGAII
            from jmetal.algorithm.multiobjective.smsemoa import SMSEMOA
            from jmetal.algorithm.multiobjective.moead import MOEAD
            from jmetal.operator.crossover import DifferentialEvolutionCrossover, SBXCrossover
            from jmetal.operator.mutation import PolynomialMutation
            from jmetal.util.aggregation_function import PenaltyBoundaryIntersection
            from jmetal.util.termination_criterion import StoppingByEvaluations
            from jmetal.problem import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
            from jmetal.problem import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7
            from jmetal.problem import WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9
            import random

            random.seed(seed)
            np.random.seed(seed)
            try:
                from jmetal.util.random_generator import PRNG

                PRNG.seed(seed)
            except Exception:
                pass

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

            archive = None
            if ALGORITHM in {"nsgaii", "nsgaii_ss", "nsgaii_archive"}:
                offspring_size = 1 if ALGORITHM == "nsgaii_ss" else POP_SIZE
                if ALGORITHM == "nsgaii_archive":
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
                    algorithm = NSGAIIArchive(
                        problem=jmetal_problem,
                        population_size=POP_SIZE,
                        offspring_population_size=offspring_size,
                        mutation=PolynomialMutation(probability=1.0 / n_var, distribution_index=MUTATION_ETA),
                        crossover=SBXCrossover(probability=CROSSOVER_PROB, distribution_index=CROSSOVER_ETA),
                        termination_criterion=StoppingByEvaluations(max_evaluations=N_EVALS),
                        archive_ref=archive,
                    )
                else:
                    algorithm = NSGAII(
                        problem=jmetal_problem,
                        population_size=POP_SIZE,
                        offspring_population_size=offspring_size,
                        mutation=PolynomialMutation(probability=1.0 / n_var, distribution_index=MUTATION_ETA),
                        crossover=SBXCrossover(probability=CROSSOVER_PROB, distribution_index=CROSSOVER_ETA),
                        termination_criterion=StoppingByEvaluations(max_evaluations=N_EVALS),
                    )
            elif ALGORITHM == "smsemoa":
                algorithm = SMSEMOA(
                    problem=jmetal_problem,
                    population_size=POP_SIZE,
                    mutation=PolynomialMutation(probability=1.0 / n_var, distribution_index=MUTATION_ETA),
                    crossover=SBXCrossover(probability=CROSSOVER_PROB, distribution_index=CROSSOVER_ETA),
                    termination_criterion=StoppingByEvaluations(max_evaluations=N_EVALS),
                )
            else:
                algorithm = MOEAD(
                    problem=jmetal_problem,
                    population_size=POP_SIZE,
                    mutation=PolynomialMutation(probability=1.0 / n_var, distribution_index=MUTATION_ETA),
                    crossover=DifferentialEvolutionCrossover(CR=MOEAD_DE_CR, F=MOEAD_DE_F, K=0.5),
                    aggregation_function=PenaltyBoundaryIntersection(n_obj, theta=MOEAD_PBI_THETA),
                    neighbourhood_selection_probability=MOEAD_DELTA,
                    max_number_of_replaced_solutions=MOEAD_REPLACE_LIMIT,
                    neighbor_size=MOEAD_NEIGHBOR_SIZE,
                    weight_files_path=MOEAD_WEIGHTS_DIR.as_posix(),
                    termination_criterion=StoppingByEvaluations(max_evaluations=N_EVALS),
                )

            start = time.perf_counter()
            algorithm.run()
            elapsed = time.perf_counter() - start

            if ALGORITHM == "nsgaii_archive" and archive is not None:
                _, F = archive.contents()
                hv = compute_hv(F, problem_name)
                igd_plus = compute_igd_plus(F, problem_name)
                n_solutions = int(F.shape[0]) if F is not None else 0
            else:
                solutions = algorithm.result()  # result() is a method, not property
                F = np.array([s.objectives for s in solutions])
                hv = compute_hv(F, problem_name)
                igd_plus = compute_igd_plus(F, problem_name)
                n_solutions = len(solutions)

            result_entry = {
                "framework": "jMetalPy",
                "problem": problem_name,
                "algorithm": ALGORITHM_DISPLAY,
                "n_evals": N_EVALS,
                "seed": seed,
                "runtime_seconds": elapsed,
                "n_solutions": n_solutions,
                "hypervolume": hv,
                "igd_plus": igd_plus,
            }
            print(f"  {problem_name} jMetalPy seed={seed}: {elapsed:.2f}s")
        except Exception as e:
            print(f"  {problem_name} jMetalPy seed={seed} FAILED: {e}")

    # Platypus
    elif framework == "platypus":
        if ALGORITHM not in {"nsgaii", "nsgaii_archive", "moead"}:
            raise ValueError("Platypus baseline is only implemented for NSGA-II variants and MOEA/D.")
        try:
            from platypus import MOEAD as PlatypusMOEAD
            from platypus import NSGAII as PlatypusNSGAII, Problem, Real
            from platypus import Archive, DifferentialEvolution, GAOperator, PM, SBX, TournamentSelector, pbi
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

            if ALGORITHM in {"nsgaii", "nsgaii_archive"}:
                archive = Archive() if ALGORITHM == "nsgaii_archive" else None
                variator = GAOperator(
                    SBX(probability=CROSSOVER_PROB, distribution_index=CROSSOVER_ETA),
                    PM(probability=1.0 / n_var, distribution_index=MUTATION_ETA),
                )
                algorithm = PlatypusNSGAII(
                    platypus_problem,
                    population_size=POP_SIZE,
                    selector=TournamentSelector(2),
                    variator=variator,
                    archive=archive,
                )
            else:

                def _weight_generator(nobjs, population_size=POP_SIZE):
                    weights = load_moead_weights(nobjs, population_size)
                    return weights.tolist()

                def _pbi_scalarization(solution, ideal_point, weights, theta=MOEAD_PBI_THETA):
                    return pbi(solution, ideal_point, weights, theta=theta)

                variator = GAOperator(
                    DifferentialEvolution(crossover_rate=MOEAD_DE_CR, step_size=MOEAD_DE_F),
                    PM(probability=1.0 / n_var, distribution_index=MUTATION_ETA),
                )
                algorithm = PlatypusMOEAD(
                    platypus_problem,
                    neighborhood_size=MOEAD_NEIGHBOR_SIZE,
                    delta=MOEAD_DELTA,
                    eta=MOEAD_REPLACE_LIMIT,
                    variator=variator,
                    weight_generator=_weight_generator,
                    scalarizing_function=_pbi_scalarization,
                    population_size=POP_SIZE,
                )

            start = time.perf_counter()
            algorithm.run(N_EVALS)
            elapsed = time.perf_counter() - start

            result_solutions = list(algorithm.result)
            F = np.array([s.objectives for s in result_solutions])
            hv = compute_hv(F, problem_name)
            igd_plus = compute_igd_plus(F, problem_name)

            result_entry = {
                "framework": "Platypus",
                "problem": problem_name,
                "algorithm": ALGORITHM_DISPLAY,
                "n_evals": N_EVALS,
                "seed": seed,
                "runtime_seconds": elapsed,
                "n_solutions": len(result_solutions),
                "hypervolume": hv,
                "igd_plus": igd_plus,
            }
            print(f"  {problem_name} Platypus seed={seed}: {elapsed:.2f}s")
        except Exception as e:
            print(f"  {problem_name} Platypus seed={seed} FAILED: {e}")

    return result_entry


def _save_partial(results_list):
    """Persist intermediate results to CSV (overwrites with latest state)."""
    filtered = [r for r in results_list if r is not None]
    if not filtered:
        return
    df_new = pd.DataFrame(filtered)
    if OUTPUT_CSV.exists() and RESUME:
        try:
            df_old = pd.read_csv(OUTPUT_CSV)
        except Exception as exc:
            print(f"Warning: failed to read existing CSV for resume ({OUTPUT_CSV}): {exc}")
            df = df_new
        else:
            df = pd.concat([df_old, df_new], ignore_index=True)
            key_cols = [c for c in ["framework", "problem", "algorithm", "n_evals", "seed"] if c in df.columns]
            if key_cols:
                df = df.drop_duplicates(subset=key_cols, keep="last")
    else:
        df = df_new
    df.to_csv(OUTPUT_CSV, index=False)


# Preflight objective alignment (guards against definition drift)
run_objective_alignment_checks()

# Build list of all jobs - split by thread-safety
PARALLEL_FRAMEWORKS = ["vamos-numpy", "vamos-numba", "vamos-moocore", "pymoo", "jmetalpy", "deap", "platypus"]
SEQUENTIAL_FRAMEWORKS = []

SCHEDULE = os.environ.get("VAMOS_PAPER_SCHEDULE", "by_job").strip().lower()
if SCHEDULE not in {"by_job", "by_framework"}:
    raise ValueError("VAMOS_PAPER_SCHEDULE must be 'by_job' or 'by_framework'.")

parallel_jobs = []
sequential_jobs = []
jobs_by_framework: dict[str, list[tuple[str, int, str]]] = {fw: [] for fw in FRAMEWORKS}

def _framework_result_name(framework: str) -> str:
    if framework.startswith("vamos-"):
        backend = framework.replace("vamos-", "")
        return f"VAMOS ({backend})"
    if framework == "jmetalpy":
        return "jMetalPy"
    if framework == "deap":
        return "DEAP"
    if framework == "platypus":
        return "Platypus"
    return framework

completed_keys: set[tuple[str, str, str, int, int]] = set()
if RESUME and OUTPUT_CSV.exists():
    try:
        existing = pd.read_csv(OUTPUT_CSV)
    except Exception as exc:
        print(f"Warning: could not load existing CSV for resume ({OUTPUT_CSV}): {exc}")
    else:
        expected_cols = {"framework", "problem", "algorithm", "n_evals", "seed"}
        if expected_cols.issubset(existing.columns):
            existing["problem"] = existing["problem"].astype(str).str.lower()
            # Only resume within the same algorithm/budget to avoid mixing runs.
            existing = existing[(existing["algorithm"] == ALGORITHM_DISPLAY) & (existing["n_evals"] == N_EVALS)]
            completed_keys = {
                (
                    str(row.framework),
                    str(row.problem).lower(),
                    str(row.algorithm),
                    int(row.n_evals),
                    int(row.seed),
                )
                for row in existing.itertuples(index=False)
            }
            print(f"Resume enabled: found {len(completed_keys)} completed runs in {OUTPUT_CSV}")
        else:
            print(f"Warning: resume is enabled but {OUTPUT_CSV} lacks required columns: {sorted(expected_cols)}")

for problem_name in PROBLEMS:
    for seed in range(N_SEEDS):
        for framework in FRAMEWORKS:
            result_fw = _framework_result_name(framework)
            if completed_keys and (result_fw, problem_name, ALGORITHM_DISPLAY, N_EVALS, seed) in completed_keys:
                continue
            job = (problem_name, seed, framework)
            jobs_by_framework[framework].append(job)
            if framework in SEQUENTIAL_FRAMEWORKS:
                sequential_jobs.append(job)
            else:
                parallel_jobs.append(job)

print(f"\nParallel jobs: {len(parallel_jobs)}")
print(f"Sequential jobs: {len(sequential_jobs)}")
print(f"Total: {len(parallel_jobs) + len(sequential_jobs)}")

# Run parallel jobs first
print(f"\nRunning {len(parallel_jobs)} parallel jobs...")
if SCHEDULE == "by_framework":
    results_list = []
    for fw in FRAMEWORKS:
        fw_jobs = jobs_by_framework.get(fw, [])
        if not fw_jobs:
            continue
        print(f"\nRunning {len(fw_jobs)} jobs for framework '{fw}'...")

        # Always run "sequential frameworks" sequentially, regardless of N_JOBS.
        if fw in SEQUENTIAL_FRAMEWORKS or N_JOBS == 1:
            bar = ProgressBar(total=len(fw_jobs), desc=f"Paper benchmark ({fw})")
            for p, s, b in fw_jobs:
                results_list.append(run_single_benchmark(p, s, b))
                if SAVE_EVERY > 0 and len(results_list) % SAVE_EVERY == 0:
                    _save_partial(results_list)
                bar.update(1)
            bar.close()
        else:
            if SAVE_EVERY > 0:
                for i in range(0, len(fw_jobs), SAVE_EVERY):
                    chunk = fw_jobs[i : i + SAVE_EVERY]
                    with joblib_progress(total=len(chunk), desc=f"Paper benchmark ({fw})"):
                        chunk_results = Parallel(n_jobs=N_JOBS, batch_size=1)(
                            delayed(run_single_benchmark)(p, s, b) for p, s, b in chunk
                        )
                    results_list.extend(chunk_results)
                    _save_partial(results_list)
            else:
                with joblib_progress(total=len(fw_jobs), desc=f"Paper benchmark ({fw})"):
                    fw_results = Parallel(n_jobs=N_JOBS, batch_size=1)(
                        delayed(run_single_benchmark)(p, s, b) for p, s, b in fw_jobs
                    )
                results_list.extend(fw_results)

        # Extra safety: flush after each framework block completes.
        _save_partial(results_list)
elif parallel_jobs:
    if N_JOBS == 1:
        bar = ProgressBar(total=len(parallel_jobs), desc="Paper benchmark")
        results_list = []
        for p, s, b in parallel_jobs:
            results_list.append(run_single_benchmark(p, s, b))
            if SAVE_EVERY > 0 and len(results_list) % SAVE_EVERY == 0:
                _save_partial(results_list)
            bar.update(1)
        bar.close()
    else:
        results_list = []
        if SAVE_EVERY > 0:
            for i in range(0, len(parallel_jobs), SAVE_EVERY):
                chunk = parallel_jobs[i : i + SAVE_EVERY]
                with joblib_progress(total=len(chunk), desc="Paper benchmark"):
                    chunk_results = Parallel(n_jobs=N_JOBS, batch_size=1)(
                        delayed(run_single_benchmark)(p, s, b) for p, s, b in chunk
                    )
                results_list.extend(chunk_results)
                _save_partial(results_list)
        else:
            with joblib_progress(total=len(parallel_jobs), desc="Paper benchmark"):
                results_list = Parallel(n_jobs=N_JOBS, batch_size=1)(
                    delayed(run_single_benchmark)(p, s, b) for p, s, b in parallel_jobs
                )
else:
    results_list = []

# Run sequential jobs (jMetalPy, Platypus)
print(f"\nRunning {len(sequential_jobs)} sequential jobs...")
seq_bar = ProgressBar(total=len(sequential_jobs), desc="Sequential jobs") if sequential_jobs else None
for p, s, b in sequential_jobs:
    result = run_single_benchmark(p, s, b)
    if result:
        results_list.append(result)
        if SAVE_EVERY > 0 and len(results_list) % SAVE_EVERY == 0:
            _save_partial(results_list)
    if seq_bar is not None:
        seq_bar.update(1)
if seq_bar is not None:
    seq_bar.close()

# Filter out None results (failed runs)
results = [r for r in results_list if r is not None]

# Save results
_save_partial(results)
print(f"\nSaved results to {OUTPUT_CSV}")
print("\nDone!")
