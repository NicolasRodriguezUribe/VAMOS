from __future__ import annotations

"""
End-to-end hyperparameter tuning example using the built-in SVM problem.

Requires the optional ``examples`` extras (scikit-learn, matplotlib).
"""

import numpy as np

from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.foundation.core.optimize import OptimizeConfig, optimize
from vamos.foundation.problem.real_world.hyperparam import HyperparameterTuningProblem


def main():
    problem = HyperparameterTuningProblem()
    cfg = (
        NSGAIIConfig()
        .pop_size(12)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", eta=20.0)
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
            termination=("n_eval", 60),
            seed=1,
            engine="numpy",
        )
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
