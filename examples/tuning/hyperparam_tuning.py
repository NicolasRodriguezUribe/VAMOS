"""
End-to-end hyperparameter tuning example using the built-in SVM problem.

Note: This example uses explicit config objects for reproducibility and tuning control.
For quick runs, prefer the unified optimize(...) API.

Usage:
    python examples/hyperparam_tuning.py

Requirements:
    pip install -e ".[examples]"  # scikit-learn, matplotlib
"""

from __future__ import annotations

import numpy as np

from vamos import optimize
from vamos.problems import HyperparameterTuningProblem
from vamos.algorithms import NSGAIIConfig


def main():
    problem = HyperparameterTuningProblem()
    cfg = (
        NSGAIIConfig.builder()
        .pop_size(12)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", pressure=2)
        .build()
    )
    result = optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=cfg,
        termination=("max_evaluations", 60),
        seed=1,
        engine="numpy",
    )
    F = result.F
    try:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.scatter(F[:, 0], F[:, 1], c=np.arange(F.shape[0]), cmap="viridis")
        plt.xlabel("Validation error")
        plt.ylabel("Complexity (proxy)")
        plt.title("Hyperparameter tuning Pareto front")
        plt.tight_layout()
        plt.show()
    except Exception:
        print("Run finished; matplotlib not available for plotting.")


if __name__ == "__main__":
    main()
