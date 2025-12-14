"""
Lightweight benchmarks for VAMOS kernels.

Run with:
  python benchmarks/bench_vamos.py
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from vamos.engine.algorithm.factory import build_algorithm
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.problem.registry import make_problem_selection


def _timeit(fn: Callable[[], None], repeat: int = 3) -> float:
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        best = min(best, end - start)
    return best


def bench_nsga2(engine: str, n: int = 1000, m: int = 2) -> float:
    selection = make_problem_selection("zdt1", n_var=30, n_obj=m)
    cfg = ExperimentConfig(population_size=n, offspring_population_size=n, max_evaluations=n * 2, seed=0)
    problem = selection.instantiate()
    algo, _ = build_algorithm("nsgaii", engine, problem, cfg, selection_pressure=2)
    rng_seed = cfg.seed
    termination = ("n_eval", cfg.max_evaluations)

    def _run():
        algo.run(problem, termination=termination, seed=rng_seed)

    return _timeit(_run)


def main():
    engines = ("numpy",)
    try:
        import numba  # noqa: F401
        engines += ("numba",)
    except ImportError:
        pass
    try:
        import moocore  # noqa: F401
        engines += ("moocore",)
    except ImportError:
        pass

    print("VAMOS kernel micro-benchmarks (lower is better)")
    for eng in engines:
        best = bench_nsga2(eng, n=200, m=2)
        print(f"{eng:<8} NSGA-II (n=200): {best*1000:.2f} ms (best-of-3)")


if __name__ == "__main__":
    main()
