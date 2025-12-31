"""
Multi-objective TSP (TSPLIB KroA100) solved with NSGA-II.

Demonstrates permutation encoding, TSPLIB loading, and Pareto front plotting.

Usage:
    python examples/tsp_tsplib_nsga2.py

Requirements:
    pip install -e ".[examples]"  # matplotlib for plotting
"""
from __future__ import annotations

import numpy as np

from vamos import (
    OptimizeConfig,
    make_problem_selection,
    optimize,
    plot_pareto_front_2d,
)
from vamos.engine.api import NSGAIIConfig


def main() -> None:
    # Load a 100-city TSPLIB instance via the public registry helper.
    selection = make_problem_selection("kroa100")
    problem = selection.instantiate()

    # NSGA-II tuned for permutation encodings (order crossover + swap mutation).
    cfg = (
        NSGAIIConfig()
        .pop_size(120)
        .offspring_size(120)
        .crossover("ox")  # order crossover for permutations
        .mutation("swap", prob="2/n")  # swap two cities with prob. 2/n
        .selection("tournament", pressure=2)
        .survival("nsga2")
        .engine("numpy")
        .fixed()
    )

    result = optimize(
        OptimizeConfig(
            problem=problem,
            algorithm="nsgaii",
            algorithm_config=cfg,
            termination=("n_eval", 15000),
            seed=7,
            engine="numpy",
        )
    )

    F = result.F
    X = result.X
    if F is None or X is None:
        raise RuntimeError("No solutions returned from the optimizer.")

    best_idx = int(np.argmin(F[:, 0]))
    print(f"Evaluated {len(F)} tours across {problem.n_var} cities.")
    print(f"Best tour length: {F[best_idx, 0]:.2f}")
    print(f"Longest edge in that tour: {F[best_idx, 1]:.2f}")
    print("First 10 cities in the best tour:", X[best_idx][:10].astype(int))

    # Pareto front: total length vs. longest edge length.
    try:
        import matplotlib.pyplot as plt

        plot_pareto_front_2d(
            F,
            labels=("Total length", "Longest edge"),
            title="TSPLIB KroA100 Pareto front (NSGA-II)",
        )
        plt.tight_layout()
        plt.show()
    except Exception as exc:  # pragma: no cover - plotting optional
        print("Plotting skipped:", exc)


if __name__ == "__main__":
    main()
