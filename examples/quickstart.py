"""
Minimal VAMOS quickstart example.

Runs NSGA-II on the ZDT1 benchmark problem and displays the Pareto front.

Usage:
    python examples/quickstart.py

Requirements:
    pip install -e .  # Core only
    pip install -e ".[examples]"  # With matplotlib for plotting
"""
from __future__ import annotations

from vamos.api import OptimizeConfig, optimize
from vamos.foundation.problems_registry import ZDT1
from vamos.engine.api import NSGAIIConfig


def main():
    # 1. Define the problem
    problem = ZDT1(n_var=30)

    # 2. Configure the algorithm
    config = (
        NSGAIIConfig()
        .pop_size(100)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .engine("numpy")
        .fixed()
    )

    # 3. Run optimization
    result = optimize(
        OptimizeConfig(
            problem=problem,
            algorithm="nsgaii",
            algorithm_config=config,
            termination=("n_eval", 10000),
            seed=42,
        )
    )

    # 4. Analyze results
    F = result.F  # Pareto front objectives
    X = result.X  # Decision variables

    print(f"Found {len(F)} Pareto-optimal solutions")
    print(f"Objective ranges:")
    print(f"  f1: [{F[:, 0].min():.4f}, {F[:, 0].max():.4f}]")
    print(f"  f2: [{F[:, 1].min():.4f}, {F[:, 1].max():.4f}]")

    # 5. Optional: Visualize
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        plt.figure(figsize=(8, 6))
        plt.scatter(F[:, 0], F[:, 1], s=30, alpha=0.7, c="steelblue", label="Found")

        # True Pareto front for ZDT1
        f1_true = np.linspace(0, 1, 100)
        f2_true = 1 - np.sqrt(f1_true)
        plt.plot(f1_true, f2_true, "k--", linewidth=2, alpha=0.5, label="True PF")

        plt.xlabel("f1")
        plt.ylabel("f2")
        plt.title("ZDT1 Pareto Front")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("\nInstall matplotlib for visualization: pip install matplotlib")


if __name__ == "__main__":
    main()
