"""
End-to-end example: multi-objective hyperparameter tuning on a real dataset.

Requirements: install with `pip install -e ".[examples]"` to pull scikit-learn/matplotlib.
"""

from __future__ import annotations

import numpy as np

from vamos.algorithm.config import NSGAIIConfig
from vamos.optimize import optimize
from vamos.problem.real_world.hyperparam import HyperparameterTuningProblem
from vamos.visualization import plot_pareto_front_2d


def build_config(pop_size: int = 24) -> dict:
    cfg = (
        NSGAIIConfig()
        .pop_size(pop_size)
        .offspring_size(pop_size)
        .crossover("sbx", prob=0.9, eta=15.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", pressure=2)
        .survival("nsga2")
        .engine("numpy")
        .result_mode("population")
        .fixed()
    )
    return cfg.to_dict()


def main(seed: int = 17) -> None:
    try:
        problem = HyperparameterTuningProblem(dataset="breast_cancer")
    except ImportError as exc:  # pragma: no cover - run-time guard when sklearn is missing
        print("Install extras with `pip install -e \".[examples]\"` to run this example.")
        print(exc)
        return

    cfg = build_config(pop_size=20)
    result = optimize(problem, cfg, termination=("n_eval", 150), seed=seed)
    F = result.F
    X = result.X
    print(f"Evaluated {F.shape[0]} candidate models.")

    # Pick a simple knee by minimizing the L1 norm of normalized objectives.
    norm = (F - F.min(axis=0)) / (F.ptp(axis=0) + 1e-8)
    knee_idx = int(np.argmin(norm.sum(axis=1)))
    print("Knee solution objectives (val. error, complexity):", F[knee_idx])
    if X is not None and hasattr(problem, "_decode_params"):
        decoded = problem._decode_params(X)  # type: ignore[attr-defined]
        print("Decoded hyperparameters at knee:", decoded[knee_idx])

    try:
        import matplotlib.pyplot as plt

        plot_pareto_front_2d(
            F,
            labels=("Validation error", "Model complexity"),
            title="Hyperparameter tuning Pareto front",
        )
        plt.tight_layout()
        plt.show()
    except Exception as exc:  # pragma: no cover - plotting optional in CI
        print("Plotting skipped:", exc)


if __name__ == "__main__":
    main()
