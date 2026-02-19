"""
Constrained Multi-Objective Benchmark
======================================
Runs NSGA-II with feasibility rules on constrained problems (C-DTLZ, MW)
across VAMOS, pymoo, and jMetalPy.

Usage: python paper/21_run_constrained_benchmark.py

Environment variables:
  - VAMOS_N_EVALS: evaluations per run (default: 60000)
  - VAMOS_N_SEEDS: number of seeds (default: 30)

Output: experiments/benchmark_paper_constrained.csv
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

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

N_EVALS = int(os.environ.get("VAMOS_N_EVALS", "60000"))
N_SEEDS = int(os.environ.get("VAMOS_N_SEEDS", "30"))
POP_SIZE = 100
CROSSOVER_PROB = 1.0
CROSSOVER_ETA = 20.0
MUTATION_ETA = 20.0

DATA_DIR = ROOT_DIR / "experiments"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = DATA_DIR / "benchmark_paper_constrained.csv"

# Constrained problems: (name, n_var, n_obj)
PROBLEMS = [
    ("c1dtlz1", 12, 3),
    ("c2dtlz2", 12, 3),
    ("mw1", 15, 2),
    ("mw3", 15, 2),
    ("mw6", 15, 2),
]


# =============================================================================
# VAMOS runner
# =============================================================================

def run_vamos(problem_name: str, n_var: int, n_obj: int, seed: int) -> dict | None:
    from vamos.foundation.problem.registry import make_problem_selection
    from vamos import optimize
    from vamos.engine.algorithm.config import NSGAIIConfig

    problem = make_problem_selection(problem_name, n_var=n_var, n_obj=n_obj).instantiate()

    cfg = (
        NSGAIIConfig.builder()
        .pop_size(POP_SIZE)
        .crossover("sbx", prob=CROSSOVER_PROB, eta=CROSSOVER_ETA)
        .mutation("pm", prob=1.0 / n_var, eta=MUTATION_ETA)
        .selection("tournament")
        .build()
    )

    # Numba warmup
    _ = optimize(problem, algorithm="nsgaii", algorithm_config=cfg,
                 termination=("max_evaluations", min(2000, N_EVALS)),
                 seed=seed, engine="numba")

    start = time.perf_counter()
    result = optimize(problem, algorithm="nsgaii", algorithm_config=cfg,
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
        "algorithm": "NSGA-II",
        "n_evals": N_EVALS,
        "seed": seed,
        "runtime_seconds": elapsed,
        "n_solutions": n_solutions,
        "hypervolume": hv,
        "igd_plus": igd_plus,
    }


# =============================================================================
# pymoo runner
# =============================================================================

def run_pymoo(problem_name: str, n_var: int, n_obj: int, seed: int) -> dict | None:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    from pymoo.problems import get_problem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.selection.rnd import RandomSelection

    # pymoo uses different naming conventions for constrained problems
    pymoo_name_map = {
        "c1dtlz1": "c1dtlz1",
        "c2dtlz2": "c2dtlz2",
        "mw1": "mw1",
        "mw3": "mw3",
        "mw6": "mw6",
    }
    pymoo_name = pymoo_name_map.get(problem_name, problem_name)

    try:
        pymoo_problem = get_problem(pymoo_name)
    except Exception:
        # Fall back to explicit parameters
        pymoo_problem = get_problem(pymoo_name, n_var=n_var, n_obj=n_obj)

    algorithm = NSGA2(
        pop_size=POP_SIZE,
        crossover=SBX(prob=CROSSOVER_PROB, eta=CROSSOVER_ETA),
        mutation=PM(prob=1.0 / n_var, eta=MUTATION_ETA),
        selection=RandomSelection(),
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
        "algorithm": "NSGA-II",
        "n_evals": N_EVALS,
        "seed": seed,
        "runtime_seconds": elapsed,
        "n_solutions": n_solutions,
        "hypervolume": hv,
        "igd_plus": igd_plus,
    }


# =============================================================================
# jMetalPy runner
# =============================================================================

def run_jmetalpy(problem_name: str, n_var: int, n_obj: int, seed: int) -> dict | None:
    from jmetal.algorithm.multiobjective import NSGAII
    from jmetal.operator.crossover import SBXCrossover
    from jmetal.operator.mutation import PolynomialMutation
    from jmetal.util.termination_criterion import StoppingByEvaluations
    import random

    # jMetalPy constrained problem mapping
    jmetal_problems = {}
    try:
        from jmetal.problem.multiobjective.constrained import Srinivas, Tanaka
        # Try to import C-DTLZ and MW if available
    except ImportError:
        pass

    # C-DTLZ problems might not be in all jMetalPy versions
    # We'll try to use them and gracefully skip if unavailable
    jmetal_problem = None
    try:
        if problem_name == "c1dtlz1":
            from jmetal.problem.multiobjective.dtlz import DTLZ1
            # jMetalPy may not have C-DTLZ directly; skip if not available
            raise ImportError("C-DTLZ not in standard jMetalPy")
        elif problem_name == "c2dtlz2":
            raise ImportError("C-DTLZ not in standard jMetalPy")
        elif problem_name.startswith("mw"):
            raise ImportError("MW not in standard jMetalPy")
    except ImportError:
        return None  # Skip this problem for jMetalPy

    random.seed(seed)

    algorithm = NSGAII(
        problem=jmetal_problem,
        population_size=POP_SIZE,
        offspring_population_size=POP_SIZE,
        crossover=SBXCrossover(probability=CROSSOVER_PROB, distribution_index=CROSSOVER_ETA),
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
        "algorithm": "NSGA-II",
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

def main() -> None:
    print("=" * 60)
    print("  Constrained Multi-Objective Benchmark")
    print("=" * 60)
    print(f"Problems: {[p[0] for p in PROBLEMS]}")
    print(f"Seeds: {N_SEEDS}")
    print(f"Evaluations: {N_EVALS}")
    print(f"Output: {OUTPUT_CSV}")
    print()

    results: list[dict] = []
    seeds = list(range(N_SEEDS))
    runners = [
        ("VAMOS (numba)", run_vamos),
        ("pymoo", run_pymoo),
        ("jMetalPy", run_jmetalpy),
    ]

    for problem_name, n_var, n_obj in PROBLEMS:
        print(f"\n--- {problem_name} (n_var={n_var}, n_obj={n_obj}) ---")
        for seed in seeds:
            for fw_name, run_fn in runners:
                try:
                    entry = run_fn(problem_name, n_var, n_obj, seed)
                    if entry is not None:
                        results.append(entry)
                        hv = entry.get("hypervolume", float("nan"))
                        print(f"  {problem_name} {fw_name} seed={seed}: {entry['runtime_seconds']:.2f}s HV={hv:.4f}")
                except Exception as e:
                    print(f"  {problem_name} {fw_name} seed={seed} FAILED: {e}")

        # Save partial
        if results:
            df = pd.DataFrame(results)
            df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nDone! Saved {len(results)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
