from __future__ import annotations

import numpy as np


def evaluate_population_with_constraints(problem, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Evaluate population and optionally return constraints G if provided by the problem.
    """
    out = {"F": np.empty((X.shape[0], problem.n_obj))}
    n_constr = getattr(problem, "n_constr", 0)
    if n_constr and n_constr > 0:
        out["G"] = np.empty((X.shape[0], n_constr))
    problem.evaluate(X, out)
    return out["F"], out.get("G")


__all__ = ["evaluate_population_with_constraints"]
