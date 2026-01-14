"""
Binary feature-selection example on a small classification dataset.

Usage:
    python examples/feature_selection.py

Requirements:
    pip install -e ".[examples]"  # scikit-learn, matplotlib
"""

from __future__ import annotations

import numpy as np

from vamos import optimize
from vamos.foundation.problem.real_world.feature_selection import FeatureSelectionProblem
from vamos.algorithms import NSGAIIConfig


def main():
    problem = FeatureSelectionProblem()
    cfg = (
        NSGAIIConfig.builder()
        .pop_size(16)
        .crossover("uniform", prob=0.9)
        .mutation("bitflip", prob="1/n")
        .selection("tournament", pressure=2)
        .build()
    )
    result = optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=cfg,
        termination=("n_eval", 80),
        seed=7,
        engine="numpy",
    )
    F = result.F
    try:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.scatter(F[:, 0], F[:, 1], c=np.arange(F.shape[0]), cmap="plasma")
        plt.xlabel("Validation error")
        plt.ylabel("Selected features")
        plt.title("Feature selection Pareto front")
        plt.tight_layout()
        plt.show()
    except Exception:
        print("Run finished; matplotlib not available for plotting.")


if __name__ == "__main__":
    main()
