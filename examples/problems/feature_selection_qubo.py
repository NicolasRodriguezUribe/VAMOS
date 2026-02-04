"""
Binary feature selection / QUBO-style pipeline on a real dataset.

Usage:
    python examples/feature_selection_qubo.py

Requirements:
    pip install -e ".[examples]"  # scikit-learn, matplotlib
"""

from __future__ import annotations

import numpy as np

from vamos import optimize
from vamos.problems import FeatureSelectionProblem
from vamos.ux.api import plot_pareto_front_2d
from vamos.algorithms import NSGAIIConfig


def build_config(pop_size: int = 30) -> NSGAIIConfig:
    cfg = (
        NSGAIIConfig.builder()
        .pop_size(pop_size)
        .offspring_size(pop_size)
        .crossover("uniform", prob=0.9)
        .mutation("bitflip", prob="1/n")
        .selection("tournament", pressure=2)
        .result_mode("population")
        .build()
    )
    return cfg


def summarize_solution(X: np.ndarray) -> str:
    idx = np.flatnonzero(X > 0.5)
    return f"{len(idx)} features -> {idx.tolist()}"


def main(seed: int = 5) -> None:
    try:
        problem = FeatureSelectionProblem(dataset="breast_cancer")
    except ImportError as exc:  # pragma: no cover
        print('Install extras: pip install -e ".[examples]"')
        print(exc)
        return

    cfg = build_config(pop_size=28)
    result = optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=cfg,
        termination=("max_evaluations", 180),
        seed=seed,
        engine="numpy",
    )
    F = result.F
    X = result.X
    print(f"Evaluated {F.shape[0]} subsets.")

    # Highlight extremes
    if X is not None:
        best_accuracy_idx = int(np.argmin(F[:, 0]))
        sparsest_idx = int(np.argmin(F[:, 1]))
        best_sol = summarize_solution(X[best_accuracy_idx])
        sparse_sol = summarize_solution(X[sparsest_idx])
        print(f"Lowest error subset: {best_sol}, obj: {F[best_accuracy_idx]}")
        print(f"Smallest subset: {sparse_sol}, obj: {F[sparsest_idx]}")

    try:
        import matplotlib.pyplot as plt

        plot_pareto_front_2d(
            F,
            labels=("Validation error", "# features"),
            title="Feature selection Pareto front",
        )
        plt.tight_layout()
        plt.show()
    except Exception as exc:  # pragma: no cover - plotting optional in CI
        print("Plotting skipped:", exc)


if __name__ == "__main__":
    main()
