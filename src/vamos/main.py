#!/usr/bin/env python3
"""VAMOS - Vectorized Architecture for Multiobjective Optimization Studies.

This is the main entry point for the VAMOS command-line interface.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add the src directory to the path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vamos.algorithm import (
    MOEAD,
    MOEADConfig,
    NSGAII,
    NSGAIIConfig,
    SMSEMOA,
    SMSEMOAConfig,
)
from vamos.kernel.numpy_backend import NumPyKernel
from vamos.problem.zdt1 import ZDT1Problem


POPULATION_SIZE = 100
MAX_EVALUATIONS = 25000
NUMBER_OF_VARIABLES = 30
SEED = 42
DECIMAL_PRECISION = 6
OUTPUT_DIR = "results/VAMOS_ZDT1"
TITLE = "VAMOS: Vectorized Architecture for Multiobjective Optimization Studies"
DEFAULT_ALGORITHM = "nsgaii"
DEFAULT_ENGINE = "numpy"
SUPPORTED_ALGORITHMS = ("nsgaii", "moead", "smsemoa")
EXPERIMENT_BACKENDS = (
    "numpy",
    "numba",
    "moocore",
    "moocore_v2",
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Vectorized multi-objective optimization demo for ZDT1."
    )
    parser.add_argument(
        "--algorithm",
        choices=(*SUPPORTED_ALGORITHMS, "both"),
        default=DEFAULT_ALGORITHM,
        help="Algorithm to run (use 'both' to execute every algorithm sequentially).",
    )
    parser.add_argument(
        "--engine",
        choices=("numpy", "numba", "moocore", "moocore_v2"),
        default=DEFAULT_ENGINE,
        help="Kernel backend to use (default: numpy).",
    )
    parser.add_argument(
        "--experiment",
        choices=("backends",),
        help="Run a predefined experiment (e.g., compare all backends).",
    )
    return parser.parse_args()


def _resolve_kernel(engine_name: str):
    """Resolve the kernel implementation based on the engine name.
    
    Args:
        engine_name: Name of the engine to use (numpy, numba, moocore, moocore_v2)
        
    Returns:
        The kernel class corresponding to the specified engine
        
    Raises:
        ValueError: If the engine name is not recognized
        SystemExit: If the required dependencies are not installed
    """
    if engine_name == "numpy":
        return NumPyKernel
    if engine_name == "numba":
        try:
            from vamos.kernel.numba_backend import NumbaKernel
        except ImportError as exc:
            raise SystemExit(
                "The 'numba' backend requires the numba package to be installed.\n"
                f"Original error: {exc}"
            ) from exc
        return NumbaKernel
    if engine_name == "moocore":
        try:
            from vamos.kernel.moocore_backend import MooCoreKernel
        except ImportError as exc:
            raise SystemExit(
                "The 'moocore' backend requires the moocore package to be installed.\n"
                f"Original error: {exc}"
            ) from exc
        return MooCoreKernel
    if engine_name == "moocore_v2":
        try:
            from vamos.kernel.moocore_backend import MooCoreKernelV2
        except ImportError as exc:
            raise SystemExit(
                "The 'moocore_v2' backend requires the moocore package to be installed.\n"
                f"Original error: {exc}"
            ) from exc
        return MooCoreKernelV2
    raise ValueError(f"Unknown engine: {engine_name}")


def _default_weight_path(problem_name: str, n_obj: int, pop_size: int) -> str:
    safe_name = problem_name.lower()
    filename = f"{safe_name}_{n_obj}obj_pop{pop_size}.csv"
    return os.path.join("build", "weights", filename)


def _build_algorithm(algorithm_name: str, engine_name: str, problem) -> object:
    kernel_cls = _resolve_kernel(engine_name)
    if algorithm_name == "nsgaii":
        cfg = (
            NSGAIIConfig()
            .pop_size(POPULATION_SIZE)
            .crossover("sbx", prob=0.9, eta=20.0)
            .mutation("pm", prob="1/n", eta=20.0)
            .selection("tournament", pressure=2)
            .survival("nsga2")
            .engine(engine_name)
        )
        return NSGAII(cfg.fixed(), kernel=kernel_cls)

    if algorithm_name == "moead":
        weight_path = _default_weight_path(
            problem.__class__.__name__, problem.n_obj, POPULATION_SIZE
        )
        cfg = (
            MOEADConfig()
            .pop_size(POPULATION_SIZE)
            .neighbor_size(min(20, POPULATION_SIZE))
            .delta(0.9)
            .replace_limit(2)
            .crossover("sbx", prob=0.9, eta=20.0)
            .mutation("pm", prob="1/n", eta=20.0)
            .aggregation("tchebycheff")
            .weight_vectors(path=weight_path)
            .engine(engine_name)
        )
        return MOEAD(cfg.fixed(), kernel=kernel_cls)

    if algorithm_name == "smsemoa":
        cfg = (
            SMSEMOAConfig()
            .pop_size(POPULATION_SIZE)
            .crossover("sbx", prob=0.9, eta=20.0)
            .mutation("pm", prob="1/n", eta=20.0)
            .selection("tournament", pressure=2)
            .reference_point(offset=0.1, adaptive=True)
            .engine(engine_name)
        )
        return SMSEMOA(cfg.fixed(), kernel=kernel_cls)

    raise ValueError(f"Unsupported algorithm '{algorithm_name}'.")


def _run_single(engine_name: str, algorithm_name: str):
    print("=" * 80)
    print(TITLE)
    print("=" * 80)

    problem = ZDT1Problem(n_var=NUMBER_OF_VARIABLES)

    algorithm = _build_algorithm(algorithm_name, engine_name, problem)

    print("Problem: ZDT1 (custom)")
    print(f"Decision variables: {problem.n_var}")
    print(f"Objectives: {problem.n_obj}")
    print(f"Algorithm: {algorithm_name.upper()}")
    print(f"Backend: {engine_name}")
    print(f"Population size: {POPULATION_SIZE}")
    print(f"Max evaluations: {MAX_EVALUATIONS}")
    print("-" * 80)

    start = time.perf_counter()
    result = algorithm.run(problem, termination=("n_eval", MAX_EVALUATIONS), seed=SEED)
    end = time.perf_counter()

    total_time_ms = (end - start) * 1000.0
    F = result["F"]

    print("Algorithm finished")
    print("-" * 80)
    print("PERFORMANCE RESULTS:")
    print(f"Total time: {total_time_ms:.2f} ms")
    print(f"Evaluations: {MAX_EVALUATIONS}")
    evals_per_sec = MAX_EVALUATIONS / max(total_time_ms / 1000.0, 1e-9)
    print(f"Evaluations/second: {evals_per_sec:.0f}")
    print(f"Final solutions: {F.shape[0]}")

    print("\nSOLUTION QUALITY:")
    obj_min = F.min(axis=0)
    obj_max = F.max(axis=0)
    for i, (mn, mx) in enumerate(zip(obj_min, obj_max), start=1):
        print(
            f"  Objective {i} range: "
            f"[{mn:.{DECIMAL_PRECISION}f}, {mx:.{DECIMAL_PRECISION}f}]"
        )

    if F.shape[1] == 2:
        spread = obj_max[0] - obj_min[0]
        print(f"  Approximate front spread in f1: {spread:.{DECIMAL_PRECISION}f}")
    else:
        spread = None

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.savetxt(os.path.join(OUTPUT_DIR, "FUN.csv"), F, delimiter=",")
    with open(os.path.join(OUTPUT_DIR, "time.txt"), "w", encoding="utf-8") as f:
        f.write(f"{total_time_ms:.2f}\n")

    print("\nResults stored in:", OUTPUT_DIR)
    print("=" * 80)

    metrics = {
        "algorithm": algorithm_name,
        "engine": engine_name,
        "time_ms": total_time_ms,
        "evals_per_sec": evals_per_sec,
        "spread": spread if F.shape[1] == 2 else None,
    }
    return metrics


def _print_summary(results):
    print("\nExperiment summary")
    print("-" * 80)
    header = (
        f"{'Algo':<8} {'Backend':<10} {'Time (ms)':>12} {'Eval/s':>12} {'Spread f1':>12}"
    )
    print(header)
    print("-" * len(header))
    for res in results:
        spread = res["spread"]
        spread_txt = f"{spread:.6f}" if spread is not None else "-"
        print(
            f"{res['algorithm']:<8} "
            f"{res['engine']:<10} "
            f"{res['time_ms']:>12.2f} "
            f"{res['evals_per_sec']:>12.0f} "
            f"{spread_txt:>12}"
        )
    print("-" * len(header))


def main():
    args = _parse_args()
    if args.experiment == "backends":
        engines = EXPERIMENT_BACKENDS
    else:
        engines = (args.engine,)
    engines = EXPERIMENT_BACKENDS    
    if args.algorithm == "both":
        algorithms = SUPPORTED_ALGORITHMS
    else:
        algorithms = (args.algorithm,)
    algorithms = SUPPORTED_ALGORITHMS
    results = []
    for engine in engines:
        for algorithm_name in algorithms:
            metrics = _run_single(engine, algorithm_name)
            results.append(metrics)

    if len(results) > 1:
        _print_summary(results)


if __name__ == "__main__":
    main()
