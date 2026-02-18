"""
NSGA-III and SPEA2 Cross-Framework Benchmark
=============================================
Runs NSGA-III (VAMOS vs pymoo) and SPEA2 (VAMOS vs jMetalPy) on the standard
benchmark suite, complementing the NSGA-II/SMS-EMOA/MOEA/D results.

Usage: python paper/20_run_additional_algorithms_benchmark.py

Environment variables:
  - VAMOS_PAPER_ALGORITHM: nsgaiii, spea2, or both (default: both)
  - VAMOS_N_EVALS: evaluations per run (default: 50000)
  - VAMOS_N_SEEDS: number of seeds (default: 30)
  - VAMOS_NUMBA_WARMUP_EVALS: warmup evaluations for Numba (default: 2000)

Output:
  - experiments/benchmark_paper_nsgaiii.csv
  - experiments/benchmark_paper_spea2.csv
"""

from __future__ import annotations

import os
import sys
import time
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

# Add local checkouts if available
DESKTOP_DIR = ROOT_DIR.parent
JMETALPY_SRC = DESKTOP_DIR / "jMetalPy" / "src"
for extra_path in (JMETALPY_SRC,):
    if extra_path.exists():
        sys.path.insert(0, str(extra_path))

try:
    from .benchmark_utils import compute_hv, compute_igd_plus
except ImportError:
    from benchmark_utils import compute_hv, compute_igd_plus

# =============================================================================
# Configuration
# =============================================================================

_algo_env = os.environ.get("VAMOS_PAPER_ALGORITHM", "both").strip().lower()
ALGORITHMS_TO_RUN = []
if _algo_env in {"nsgaiii", "nsga3", "nsga-iii"}:
    ALGORITHMS_TO_RUN = ["nsgaiii"]
elif _algo_env in {"spea2"}:
    ALGORITHMS_TO_RUN = ["spea2"]
else:
    ALGORITHMS_TO_RUN = ["nsgaiii", "spea2"]

N_EVALS = int(os.environ.get("VAMOS_N_EVALS", "50000"))
N_SEEDS = int(os.environ.get("VAMOS_N_SEEDS", "30"))
NUMBA_WARMUP_EVALS = int(os.environ.get("VAMOS_NUMBA_WARMUP_EVALS", "2000"))

DATA_DIR = ROOT_DIR / "experiments"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Problem dimensions (matching 01_run_paper_benchmark.py)
ZDT_N_VAR = {"zdt1": 30, "zdt2": 30, "zdt3": 30, "zdt4": 10, "zdt6": 10}
DTLZ_N_VAR = {"dtlz1": 7, "dtlz2": 12, "dtlz3": 12, "dtlz4": 12, "dtlz5": 12, "dtlz6": 12, "dtlz7": 22}
WFG_N_VAR = 24
ZDT_N_OBJ = 2
DTLZ_N_OBJ = 3
WFG_N_OBJ = 2

# Operator parameters (matching NSGA-II benchmark)
POP_SIZE = 100
CROSSOVER_PROB = 1.0
CROSSOVER_ETA_NSGAIII = 30.0  # NSGA-III typically uses eta=30
CROSSOVER_ETA_SPEA2 = 20.0
MUTATION_ETA = 20.0

# ZDT (2-obj) + DTLZ (3-obj) problems
PROBLEMS = list(ZDT_N_VAR.keys()) + list(DTLZ_N_VAR.keys()) + [f"wfg{i}" for i in range(1, 10)]


def get_problem_dims(name: str) -> tuple[int, int]:
    if name in ZDT_N_VAR:
        return ZDT_N_VAR[name], ZDT_N_OBJ
    if name in DTLZ_N_VAR:
        return DTLZ_N_VAR[name], DTLZ_N_OBJ
    if name.startswith("wfg"):
        return WFG_N_VAR, WFG_N_OBJ
    raise ValueError(f"Unknown problem: {name}")


def nsgaiii_pop_size(n_obj: int, divisions: int = 12) -> int:
    """Das-Dennis reference directions count = C(divisions + n_obj - 1, n_obj - 1)."""
    return comb(divisions + n_obj - 1, n_obj - 1)


# =============================================================================
# NSGA-III runners
# =============================================================================

def run_vamos_nsgaiii(problem_name: str, seed: int) -> dict | None:
    from vamos.foundation.problem.registry import make_problem_selection
    from vamos import optimize
    from vamos.engine.algorithm.config.nsgaiii import NSGAIIIConfig

    n_var, n_obj = get_problem_dims(problem_name)
    problem = make_problem_selection(problem_name, n_var=n_var, n_obj=n_obj).instantiate()

    divisions = 12 if n_obj == 3 else 6
    pop = nsgaiii_pop_size(n_obj, divisions)

    cfg = (
        NSGAIIIConfig.builder()
        .pop_size(pop)
        .crossover("sbx", prob=CROSSOVER_PROB, eta=CROSSOVER_ETA_NSGAIII)
        .mutation("pm", prob=1.0 / n_var, eta=MUTATION_ETA)
        .selection("tournament")
        .reference_directions(divisions=divisions)
        .pop_size_auto(True)
        .build()
    )

    # Numba warmup
    if NUMBA_WARMUP_EVALS > 0:
        _ = optimize(problem, algorithm="nsgaiii", algorithm_config=cfg,
                     termination=("max_evaluations", min(NUMBA_WARMUP_EVALS, N_EVALS)),
                     seed=seed, engine="numba")

    start = time.perf_counter()
    result = optimize(problem, algorithm="nsgaiii", algorithm_config=cfg,
                      termination=("max_evaluations", N_EVALS),
                      seed=seed, engine="numba")
    elapsed = time.perf_counter() - start

    F = result.F
    n_solutions = result.X.shape[0] if result.X is not None else 0
    hv = compute_hv(F, problem_name) if F is not None else float("nan")
    igd_plus = compute_igd_plus(F, problem_name) if F is not None else float("nan")

    return {
        "framework": "VAMOS (numba)",
        "problem": problem_name,
        "algorithm": "NSGA-III",
        "n_evals": N_EVALS,
        "seed": seed,
        "runtime_seconds": elapsed,
        "n_solutions": n_solutions,
        "hypervolume": hv,
        "igd_plus": igd_plus,
    }


def run_pymoo_nsgaiii(problem_name: str, seed: int) -> dict | None:
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    from pymoo.problems import get_problem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM

    n_var, n_obj = get_problem_dims(problem_name)
    pymoo_problem = get_problem(problem_name)

    divisions = 12 if n_obj == 3 else 6
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=divisions)

    algorithm = NSGA3(
        ref_dirs=ref_dirs,
        pop_size=len(ref_dirs),
        crossover=SBX(prob=CROSSOVER_PROB, eta=CROSSOVER_ETA_NSGAIII),
        mutation=PM(prob=1.0 / n_var, eta=MUTATION_ETA),
    )

    start = time.perf_counter()
    res = minimize(pymoo_problem, algorithm, get_termination("n_eval", N_EVALS),
                   seed=seed, verbose=False)
    elapsed = time.perf_counter() - start

    F = res.F
    n_solutions = res.X.shape[0] if res.X is not None else 0
    hv = compute_hv(F, problem_name) if F is not None else float("nan")
    igd_plus = compute_igd_plus(F, problem_name) if F is not None else float("nan")

    return {
        "framework": "pymoo",
        "problem": problem_name,
        "algorithm": "NSGA-III",
        "n_evals": N_EVALS,
        "seed": seed,
        "runtime_seconds": elapsed,
        "n_solutions": n_solutions,
        "hypervolume": hv,
        "igd_plus": igd_plus,
    }


# =============================================================================
# SPEA2 runners
# =============================================================================

def run_vamos_spea2(problem_name: str, seed: int) -> dict | None:
    from vamos.foundation.problem.registry import make_problem_selection
    from vamos import optimize
    from vamos.engine.algorithm.config.spea2 import SPEA2Config

    n_var, n_obj = get_problem_dims(problem_name)
    problem = make_problem_selection(problem_name, n_var=n_var, n_obj=n_obj).instantiate()

    cfg = (
        SPEA2Config.builder()
        .pop_size(POP_SIZE)
        .archive_size(POP_SIZE)
        .crossover("sbx", prob=CROSSOVER_PROB, eta=CROSSOVER_ETA_SPEA2)
        .mutation("pm", prob=1.0 / n_var, eta=MUTATION_ETA)
        .selection("tournament")
        .build()
    )

    # Numba warmup
    if NUMBA_WARMUP_EVALS > 0:
        _ = optimize(problem, algorithm="spea2", algorithm_config=cfg,
                     termination=("max_evaluations", min(NUMBA_WARMUP_EVALS, N_EVALS)),
                     seed=seed, engine="numba")

    start = time.perf_counter()
    result = optimize(problem, algorithm="spea2", algorithm_config=cfg,
                      termination=("max_evaluations", N_EVALS),
                      seed=seed, engine="numba")
    elapsed = time.perf_counter() - start

    F = result.F
    n_solutions = result.X.shape[0] if result.X is not None else 0
    hv = compute_hv(F, problem_name) if F is not None else float("nan")
    igd_plus = compute_igd_plus(F, problem_name) if F is not None else float("nan")

    return {
        "framework": "VAMOS (numba)",
        "problem": problem_name,
        "algorithm": "SPEA2",
        "n_evals": N_EVALS,
        "seed": seed,
        "runtime_seconds": elapsed,
        "n_solutions": n_solutions,
        "hypervolume": hv,
        "igd_plus": igd_plus,
    }


def run_jmetalpy_spea2(problem_name: str, seed: int) -> dict | None:
    from jmetal.algorithm.multiobjective.spea2 import SPEA2
    from jmetal.operator.crossover import SBXCrossover
    from jmetal.operator.mutation import PolynomialMutation
    from jmetal.util.termination_criterion import StoppingByEvaluations

    n_var, n_obj = get_problem_dims(problem_name)
    wfg_k = 4 if WFG_N_OBJ == 2 else 2 * (WFG_N_OBJ - 1)
    wfg_l = WFG_N_VAR - wfg_k

    # Import jMetalPy problems
    from jmetal.problem.multiobjective.zdt import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
    from jmetal.problem.multiobjective.dtlz import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7
    from jmetal.problem.multiobjective.wfg import WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9

    jmetal_problems = {
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

    jmetal_problem = jmetal_problems.get(problem_name)
    if jmetal_problem is None:
        return None

    import random
    random.seed(seed)

    algorithm = SPEA2(
        problem=jmetal_problem,
        population_size=POP_SIZE,
        offspring_population_size=POP_SIZE,
        crossover=SBXCrossover(probability=CROSSOVER_PROB, distribution_index=CROSSOVER_ETA_SPEA2),
        mutation=PolynomialMutation(probability=1.0 / n_var, distribution_index=MUTATION_ETA),
        termination_criterion=StoppingByEvaluations(max_evaluations=N_EVALS),
    )

    start = time.perf_counter()
    algorithm.run()
    elapsed = time.perf_counter() - start

    solutions = algorithm.result()
    F = np.array([s.objectives for s in solutions])
    n_solutions = len(solutions)
    hv = compute_hv(F, problem_name)
    igd_plus = compute_igd_plus(F, problem_name)

    return {
        "framework": "jMetalPy",
        "problem": problem_name,
        "algorithm": "SPEA2",
        "n_evals": N_EVALS,
        "seed": seed,
        "runtime_seconds": elapsed,
        "n_solutions": n_solutions,
        "hypervolume": hv,
        "igd_plus": igd_plus,
    }


# =============================================================================
# Main
# =============================================================================

def run_benchmark(algo_name: str, runners: list[tuple[str, callable]], output_csv: Path) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {algo_name} Benchmark")
    print(f"{'=' * 60}")
    print(f"Problems: {len(PROBLEMS)}")
    print(f"Seeds: {N_SEEDS}")
    print(f"Evaluations: {N_EVALS}")
    print(f"Frameworks: {[name for name, _ in runners]}")
    print(f"Output: {output_csv}")
    print()

    results: list[dict] = []
    seeds = list(range(N_SEEDS))

    for problem_name in PROBLEMS:
        n_var, n_obj = get_problem_dims(problem_name)
        for seed in seeds:
            for fw_name, run_fn in runners:
                try:
                    entry = run_fn(problem_name, seed)
                    if entry is not None:
                        results.append(entry)
                        hv = entry.get("hypervolume", float("nan"))
                        print(f"  {problem_name} {fw_name} seed={seed}: {entry['runtime_seconds']:.2f}s HV={hv:.4f}")
                except Exception as e:
                    print(f"  {problem_name} {fw_name} seed={seed} FAILED: {e}")

        # Save partial results after each problem
        if results:
            df = pd.DataFrame(results)
            df.to_csv(output_csv, index=False)

    print(f"\nSaved {len(results)} rows to {output_csv}")


def main() -> None:
    if "nsgaiii" in ALGORITHMS_TO_RUN:
        runners = [
            ("VAMOS (numba)", run_vamos_nsgaiii),
            ("pymoo", run_pymoo_nsgaiii),
        ]
        run_benchmark("NSGA-III", runners, DATA_DIR / "benchmark_paper_nsgaiii.csv")

    if "spea2" in ALGORITHMS_TO_RUN:
        runners = [
            ("VAMOS (numba)", run_vamos_spea2),
            ("jMetalPy", run_jmetalpy_spea2),
        ]
        run_benchmark("SPEA2", runners, DATA_DIR / "benchmark_paper_spea2.csv")


if __name__ == "__main__":
    main()
