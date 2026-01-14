"""
Custom two-objective problem defined inline and solved with NSGA-II.

Shows the minimal problem interface (n_var, n_obj, xl, xu, encoding, evaluate)
and how to plug it into `vamos.optimize()` directly using the Unified API.

Usage:
    python examples/custom_problem_definition.py
"""

from __future__ import annotations

import numpy as np
from vamos import optimize


class CustomBiObjectiveProblem:
    """
    Simple convex/concave two-objective toy problem.

    Decision variables:
        x0, x1 in [0, 1]
    Objectives (minimize):
        f1 = x0
        f2 = (1 + x1) * (1 - sqrt(x0)) + 0.1 * sin(5 * x0)
    """

    def __init__(self) -> None:
        self.n_var = 2
        self.n_obj = 2
        self.xl = np.array([0.0, 0.0], dtype=float)
        self.xu = np.array([1.0, 1.0], dtype=float)
        self.encoding = "real"

    def evaluate(self, X: np.ndarray, out: dict) -> None:
        X = np.asarray(X, dtype=float)
        # Vectorized evaluation for performance
        f1 = X[:, 0]
        g = 1.0 + X[:, 1]
        f2 = g * (1.0 - np.sqrt(np.clip(X[:, 0], 0.0, 1.0))) + 0.1 * np.sin(5.0 * X[:, 0])

        # Write directly to output buffers
        F = out["F"]
        F[:, 0] = f1
        F[:, 1] = f2


def main() -> None:
    # 1. Instantiate existing custom class
    problem = CustomBiObjectiveProblem()

    # 2. Run optimize directy with the instance
    # Note: Unified API handles instance dispatch automatically
    print("optimizing custom problem instance...")
    result = optimize(problem, algorithm="nsgaii", budget=4000, seed=3, verbose=True)

    F = result.F
    print(f"\nSolutions: {len(F)}")
    print(f"Objective ranges: f1=[{F[:, 0].min():.3f}, {F[:, 0].max():.3f}], f2=[{F[:, 1].min():.3f}, {F[:, 1].max():.3f}]")

    # 3. Use built-in explorer if dependencies available (plotly)
    try:
        # explore_result_front(result, title="Custom Problem Result")
        pass  # Commented out to avoid auto-launching in non-interactive run
    except:
        pass


if __name__ == "__main__":
    main()
