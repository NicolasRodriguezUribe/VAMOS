"""
Memory Consumption Benchmark
=============================
Measures peak memory usage during NSGA-II optimization on representative
problems, comparing VAMOS (Numba) vs pymoo vs jMetalPy.

Uses Python's tracemalloc for peak memory measurement.

Usage: python paper/23_run_memory_benchmark.py

Environment variables:
  - VAMOS_N_EVALS: evaluations per run (default: 50000)
  - VAMOS_N_SEEDS: number of seeds (default: 5)

Output: experiments/memory_benchmark.csv
"""

from __future__ import annotations

import os
import sys
import time
import tracemalloc
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

# =============================================================================
# Configuration
# =============================================================================

N_EVALS = int(os.environ.get("VAMOS_N_EVALS", "50000"))
N_SEEDS = int(os.environ.get("VAMOS_N_SEEDS", "5"))
POP_SIZE = 100
CROSSOVER_PROB = 1.0
CROSSOVER_ETA = 20.0
MUTATION_ETA = 20.0

DATA_DIR = ROOT_DIR / "experiments"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = DATA_DIR / "memory_benchmark.csv"

# Representative problems
PROBLEMS = [
    ("zdt1", 30, 2),
    ("dtlz2", 12, 3),
    ("wfg4", 24, 2),
]


# =============================================================================
# Memory-profiled runners
# =============================================================================

def measure_peak_memory(run_fn, *args, **kwargs) -> tuple[float, float]:
    """Run a function and return (peak_memory_mb, runtime_seconds)."""
    import gc
    gc.collect()

    tracemalloc.start()
    start = time.perf_counter()
    try:
        run_fn(*args, **kwargs)
    finally:
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    return peak / (1024 * 1024), elapsed  # Convert bytes to MB


def run_vamos(problem_name: str, n_var: int, n_obj: int, seed: int):
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
    optimize(problem, algorithm="nsgaii", algorithm_config=cfg,
             termination=("max_evaluations", N_EVALS),
             seed=seed, engine="numba")


def run_pymoo(problem_name: str, n_var: int, n_obj: int, seed: int):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    from pymoo.problems import get_problem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.selection.rnd import RandomSelection

    pymoo_problem = get_problem(problem_name)
    algorithm = NSGA2(
        pop_size=POP_SIZE,
        crossover=SBX(prob=CROSSOVER_PROB, eta=CROSSOVER_ETA),
        mutation=PM(prob=1.0 / n_var, eta=MUTATION_ETA),
        selection=RandomSelection(),
    )
    minimize(pymoo_problem, algorithm, get_termination("n_eval", N_EVALS),
             seed=seed, verbose=False)


def run_jmetalpy(problem_name: str, n_var: int, n_obj: int, seed: int):
    from jmetal.algorithm.multiobjective import NSGAII
    from jmetal.operator.crossover import SBXCrossover
    from jmetal.operator.mutation import PolynomialMutation
    from jmetal.util.termination_criterion import StoppingByEvaluations
    from jmetal.problem.multiobjective.zdt import ZDT1
    from jmetal.problem.multiobjective.dtlz import DTLZ2
    from jmetal.problem.multiobjective.wfg import WFG4
    import random

    jmetal_problems = {
        "zdt1": ZDT1(number_of_variables=30),
        "dtlz2": DTLZ2(number_of_variables=12, number_of_objectives=3),
        "wfg4": WFG4(number_of_variables=24, number_of_objectives=2, k=4, l=20),
    }
    jmetal_problem = jmetal_problems.get(problem_name)
    if jmetal_problem is None:
        raise ValueError(f"Problem {problem_name} not configured for jMetalPy")

    random.seed(seed)
    algorithm = NSGAII(
        problem=jmetal_problem,
        population_size=POP_SIZE,
        offspring_population_size=POP_SIZE,
        crossover=SBXCrossover(probability=CROSSOVER_PROB, distribution_index=CROSSOVER_ETA),
        mutation=PolynomialMutation(probability=1.0 / n_var, distribution_index=MUTATION_ETA),
        termination_criterion=StoppingByEvaluations(max_evaluations=N_EVALS),
    )
    algorithm.run()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 60)
    print("  Memory Consumption Benchmark")
    print("=" * 60)
    print(f"Problems: {[p[0] for p in PROBLEMS]}")
    print(f"Seeds: {N_SEEDS}")
    print(f"Evaluations: {N_EVALS}")
    print(f"Output: {OUTPUT_CSV}")
    print()

    runners = [
        ("VAMOS (numba)", run_vamos),
        ("pymoo", run_pymoo),
        ("jMetalPy", run_jmetalpy),
    ]

    results: list[dict] = []
    seeds = list(range(N_SEEDS))

    # Numba warmup (JIT compilation adds memory that shouldn't count)
    print("Numba warmup...")
    try:
        run_vamos("zdt1", 30, 2, 0)
    except Exception:
        pass
    print()

    for problem_name, n_var, n_obj in PROBLEMS:
        print(f"--- {problem_name} ---")
        for seed in seeds:
            for fw_name, run_fn in runners:
                try:
                    peak_mb, elapsed = measure_peak_memory(run_fn, problem_name, n_var, n_obj, seed)
                    results.append({
                        "framework": fw_name,
                        "problem": problem_name,
                        "seed": seed,
                        "n_evals": N_EVALS,
                        "peak_memory_mb": peak_mb,
                        "runtime_seconds": elapsed,
                    })
                    print(f"  {fw_name} seed={seed}: {peak_mb:.1f} MB, {elapsed:.2f}s")
                except Exception as e:
                    print(f"  {fw_name} seed={seed} FAILED: {e}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved {len(results)} rows to {OUTPUT_CSV}")

        # Summary
        print("\nPeak Memory Summary (MB):")
        summary = df.groupby(["framework", "problem"])["peak_memory_mb"].median()
        print(summary.unstack().to_string())


if __name__ == "__main__":
    main()
