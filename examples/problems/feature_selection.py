"""
Binary feature-selection example on a small classification dataset.

Usage:
    python examples/feature_selection.py

Requirements:
    pip install -e ".[examples]"  # scikit-learn, matplotlib
"""

from __future__ import annotations

import numpy as np

from vamos.api import OptimizeConfig, optimize
from vamos.foundation.problems_registry import FeatureSelectionProblem
from vamos.engine.api import NSGAIIConfig


def main():
    problem = FeatureSelectionProblem()
    cfg = (
        NSGAIIConfig()
        .pop_size(16)
        .crossover("uniform", prob=0.9)
        .mutation("bitflip", prob="1/n")
        .selection("tournament", pressure=2)
        .fixed()
    )
    result = optimize(
        OptimizeConfig(
            problem=problem,
            algorithm="nsgaii",
            algorithm_config=cfg,
            termination=("n_eval", 80),
            seed=7,
            engine="numpy",
        )
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
