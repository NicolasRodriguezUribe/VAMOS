from __future__ import annotations

from typing import Any

import numpy as np


def evaluate_population_with_constraints(problem: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Evaluate population and optionally return constraints G if provided by the problem.
    """
    out = {"F": np.empty((X.shape[0], problem.n_obj))}
    n_constraints = getattr(problem, "n_constraints", 0)
    if n_constraints and n_constraints > 0:
        out["G"] = np.empty((X.shape[0], n_constraints))
    problem.evaluate(X, out)
    return out["F"], out.get("G")


__all__ = ["evaluate_population_with_constraints"]
