"""
Custom two-objective problem defined inline and solved with NSGA-II.

Shows the minimal problem interface (n_var, n_obj, xl, xu, encoding, evaluate)
and how to plug it into `optimize()` using only the public API.

Usage:
    python examples/custom_problem_definition.py

Requirements:
    pip install -e ".[examples]"  # matplotlib for plotting
"""
from __future__ import annotations

import numpy as np

from vamos.api import OptimizeConfig, optimize
from vamos.ux.api import plot_pareto_front_2d
from vamos.engine.api import NSGAIIConfig


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
        f1 = X[:, 0]
        g = 1.0 + X[:, 1]
        f2 = g * (1.0 - np.sqrt(np.clip(X[:, 0], 0.0, 1.0))) + 0.1 * np.sin(5.0 * X[:, 0])
        F = out["F"]
        F[:, 0] = f1
        F[:, 1] = f2


def main() -> None:
    problem = CustomBiObjectiveProblem()
    cfg = (
        NSGAIIConfig()
        .pop_size(60)
        .offspring_size(60)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob="1/n", eta=20.0)
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
            termination=("n_eval", 4000),
            seed=3,
            engine="numpy",
        )
    )

    F = result.F
    print(f"Solutions: {len(F)}, objective ranges f1=[{F[:,0].min():.3f}, {F[:,0].max():.3f}], "
          f"f2=[{F[:,1].min():.3f}, {F[:,1].max():.3f}]")
    knee = result.best("knee")
    print("Knee candidate objectives:", knee["F"])
    if knee["X"] is not None:
        print("Knee decision variables:", knee["X"])

    try:
        import matplotlib.pyplot as plt

        plot_pareto_front_2d(
            F,
            labels=("f1 = x0", "f2 (shaped by x1)"),
            title="Custom bi-objective Pareto front",
        )
        plt.tight_layout()
        plt.show()
    except Exception as exc:  # pragma: no cover - plotting optional
        print("Plotting skipped:", exc)


if __name__ == "__main__":
    main()
