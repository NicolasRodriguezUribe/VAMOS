"""
Minimal VAMOS quickstart example.

Runs NSGA-II on the ZDT1 benchmark problem and displays the Pareto front.
Uses the Unified API (`vamos.optimize`) for conciseness.
For a no-code option, try the CLI wizard: `vamos quickstart`.
Use `vamos quickstart --template list` to explore domain templates.

Usage:
    python examples/quickstart.py

Requirements:
    pip install -e .  # Core only
    pip install -e ".[analysis]"  # With matplotlib for plotting
"""

from __future__ import annotations
from vamos import optimize


def main():
    # 1. Run optimization with a single improved command
    # - "zdt1": Standard benchmark
    # - "nsgaii": Standard algorithm
    # - budget: Stopping criterion
    print("Running NSGA-II on ZDT1...")
    result = optimize("zdt1", algorithm="nsgaii", budget=5000, seed=42)

    # 2. Analyze results
    F = result.F  # Pareto front objectives
    print(f"\nFound {len(F)} Pareto-optimal solutions")
    print(f"Objective ranges:")
    print(f"  f1: [{F[:, 0].min():.4f}, {F[:, 0].max():.4f}]")
    print(f"  f2: [{F[:, 1].min():.4f}, {F[:, 1].max():.4f}]")

    # 3. Optional: Visualize
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
