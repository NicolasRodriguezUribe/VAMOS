from __future__ import annotations

"""
Binary feature-selection example on a small classification dataset.

Requires the optional ``examples`` extras (scikit-learn, matplotlib).
"""

import numpy as np

from vamos.algorithm.config import NSGAIIConfig
from vamos.optimize import optimize
from vamos.problem.real_world.feature_selection import FeatureSelectionProblem


def main():
    problem = FeatureSelectionProblem()
    cfg = (
        NSGAIIConfig()
        .pop_size(16)
        .crossover("uniform", prob=0.9)
        .mutation("bit_flip", prob="1/n")
        .selection("tournament", pressure=2)
        .survival("nsga2")
        .engine("numpy")
        .fixed()
    )
    result = optimize(problem, cfg, termination=("n_eval", 80), seed=7)
    F = result.F if isinstance(result, dict) else result.F
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
