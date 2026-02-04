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

from vamos import optimize
from vamos.problems import WeldedBeamDesignProblem
from vamos.algorithms import NSGAIIConfig


def main():
    problem = WeldedBeamDesignProblem()
    cfg = (
        NSGAIIConfig.builder()
        .pop_size(20)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", pressure=2)
        .constraint_mode("feasibility")
        .build()
    )
    result = optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=cfg,
        termination=("max_evaluations", 100),
        seed=3,
        engine="numpy",
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
