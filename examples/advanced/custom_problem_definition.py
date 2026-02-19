"""
Custom problem definition using the Problem base class.

Shows both the unconstrained and constrained cases, and how to plug them
into `vamos.optimize()` directly using the Unified API.

Usage:
    python examples/advanced/custom_problem_definition.py
"""

from __future__ import annotations

import numpy as np
from vamos import Problem, optimize


# ---------------------------------------------------------------------------
# Unconstrained problem — override objectives() only
# ---------------------------------------------------------------------------

class CustomBiObjectiveProblem(Problem):
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

    def objectives(self, X: np.ndarray) -> np.ndarray:
        # X: (N, 2) batch of candidate solutions
        f1 = X[:, 0]
        g = 1.0 + X[:, 1]
        f2 = g * (1.0 - np.sqrt(np.clip(X[:, 0], 0.0, 1.0))) + 0.1 * np.sin(5.0 * X[:, 0])
        return np.column_stack([f1, f2])


# ---------------------------------------------------------------------------
# Constrained problem — override objectives() and constraints()
# ---------------------------------------------------------------------------

class ConstrainedBiObjectiveProblem(Problem):
    """
    Two-objective problem with one inequality constraint.

    Decision variables: x0, x1, x2 in [0, 1]
    Objectives (minimize):
        f1 = sum(x^2)
        f2 = sum((x - 1)^2)
    Constraint (g <= 0 means feasible):
        g = sum(x) - 2  (i.e. sum(x) <= 2)
    """

    n_constraints = 1  # declare at class level

    def __init__(self) -> None:
        self.n_var = 3
        self.n_obj = 2
        self.xl = np.zeros(3)
        self.xu = np.ones(3)

    def objectives(self, X: np.ndarray) -> np.ndarray:
        f1 = np.sum(X ** 2, axis=1)
        f2 = np.sum((X - 1) ** 2, axis=1)
        return np.column_stack([f1, f2])

    def constraints(self, X: np.ndarray) -> np.ndarray:
        # Sign convention: g(x) <= 0 means feasible (negative = OK).
        g = np.sum(X, axis=1) - 2.0
        return g.reshape(-1, 1)


def main() -> None:
    # --- Unconstrained ---
    problem = CustomBiObjectiveProblem()
    print("Optimizing unconstrained custom problem...")
    result = optimize(problem, algorithm="nsgaii", max_evaluations=4000, seed=3, verbose=True)

    F = result.F
    print(f"\nSolutions: {len(F)}")
    print(f"Objective ranges: f1=[{F[:, 0].min():.3f}, {F[:, 0].max():.3f}], "
          f"f2=[{F[:, 1].min():.3f}, {F[:, 1].max():.3f}]")

    # --- Constrained ---
    constrained = ConstrainedBiObjectiveProblem()
    print("\nOptimizing constrained custom problem...")
    result_c = optimize(constrained, algorithm="nsgaii", max_evaluations=4000, seed=3, verbose=True)

    F_c = result_c.F
    print(f"\nConstrained solutions: {len(F_c)}")
    print(f"Objective ranges: f1=[{F_c[:, 0].min():.3f}, {F_c[:, 0].max():.3f}], "
          f"f2=[{F_c[:, 1].min():.3f}, {F_c[:, 1].max():.3f}]")


if __name__ == "__main__":
    main()
