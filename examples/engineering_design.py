from __future__ import annotations

"""
Mixed-encoding welded beam design example.

Demonstrates constraints and mixed decision variables.
"""

import numpy as np

from vamos.algorithm.config import NSGAIIConfig
from vamos.core.optimize import optimize
from vamos.problem.real_world.engineering import WeldedBeamDesignProblem


def main():
    problem = WeldedBeamDesignProblem()
    cfg = (
        NSGAIIConfig()
        .pop_size(20)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", eta=20.0)
        .selection("tournament", pressure=2)
        .survival("nsga2")
        .engine("numpy")
        .constraint_mode("feasibility")
        .fixed()
    )
    result = optimize(problem, cfg, termination=("n_eval", 100), seed=3)
    F = result.F if isinstance(result, dict) else result.F
    try:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.scatter(F[:, 0], F[:, 1], c=np.arange(F.shape[0]), cmap="magma")
        plt.xlabel("Fabrication cost")
        plt.ylabel("Deflection proxy")
        plt.title("Welded beam design Pareto front")
        plt.tight_layout()
        plt.show()
    except Exception:
        print("Run finished; matplotlib not available for plotting.")


if __name__ == "__main__":
    main()
