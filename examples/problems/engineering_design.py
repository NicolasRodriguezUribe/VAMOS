"""
Mixed-encoding welded beam design example.

Demonstrates constraints and mixed decision variables.

Usage:
    python examples/engineering_design.py

Requirements:
    pip install -e ".[examples]"  # matplotlib (optional)
"""

from __future__ import annotations

import numpy as np

from vamos.api import OptimizeConfig, optimize
from vamos.foundation.problems_registry import WeldedBeamDesignProblem
from vamos.engine.api import NSGAIIConfig


def main():
    problem = WeldedBeamDesignProblem()
    cfg = (
        NSGAIIConfig()
        .pop_size(20)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", pressure=2)
        .engine("numpy")
        .constraint_mode("feasibility")
        .fixed()
    )
    result = optimize(
        OptimizeConfig(
            problem=problem,
            algorithm="nsgaii",
            algorithm_config=cfg,
            termination=("n_eval", 100),
            seed=3,
            engine="numpy",
        )
    )
    F = result.F
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
