"""
Binary feature selection / QUBO-style pipeline on a real dataset.

Requirements: `pip install -e ".[examples]"` for scikit-learn + plotting.
"""

from __future__ import annotations

import numpy as np

from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.foundation.core.optimize import optimize
from vamos.foundation.problem.real_world.feature_selection import FeatureSelectionProblem
from vamos.ux.visualization import plot_pareto_front_2d


def build_config(pop_size: int = 30) -> dict:
    cfg = (
        NSGAIIConfig()
        .pop_size(pop_size)
        .offspring_size(pop_size)
        .crossover("uniform", prob=0.9)
        .mutation("bitflip", prob="1/n")
        .selection("tournament", pressure=2)
        .survival("nsga2")
        .engine("numpy")
        .result_mode("population")
        .fixed()
    )
    return cfg.to_dict()


def summarize_solution(X: np.ndarray) -> str:
    idx = np.flatnonzero(X > 0.5)
    return f"{len(idx)} features -> {idx.tolist()}"


def main(seed: int = 5) -> None:
    try:
        problem = FeatureSelectionProblem(dataset="breast_cancer")
    except ImportError as exc:  # pragma: no cover - run-time guard when sklearn missing
        print("Install extras with `pip install -e \".[examples]\"` to run this example.")
        print(exc)
        return

    cfg = build_config(pop_size=28)
    result = optimize(problem, cfg, termination=("n_eval", 180), seed=seed)
    F = result.F
    X = result.X
    print(f"Evaluated {F.shape[0]} subsets.")

    # Highlight extremes
    if X is not None:
        best_accuracy_idx = int(np.argmin(F[:, 0]))
        sparsest_idx = int(np.argmin(F[:, 1]))
        print("Lowest error subset:", summarize_solution(X[best_accuracy_idx]), "obj:", F[best_accuracy_idx])
        print("Smallest subset:", summarize_solution(X[sparsest_idx]), "obj:", F[sparsest_idx])

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
