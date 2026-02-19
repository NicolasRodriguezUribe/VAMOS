"""
Utility helpers for constraint handling.
"""

from __future__ import annotations

import numpy as np


def compute_violation(G: np.ndarray | None, *, n: int | None = None) -> np.ndarray:
    """Sum of positive parts per-solution; assumes G shape (N, n_constr), g<=0 satisfied.

    When *G* is ``None`` (unconstrained), returns an array of zeros.  If *n* is
    given the array has length *n*; otherwise it has length 0 for backward
    compatibility.
    """
    if G is None:
        return np.zeros(n or 0, dtype=float)
    positive = np.maximum(G, 0.0)
    return np.asarray(np.sum(positive, axis=1), dtype=float)


def is_feasible(G: np.ndarray | None, *, n: int | None = None, eps: float = 0.0) -> np.ndarray:
    """Boolean feasibility mask; assumes G shape (N, n_constr).

    When *G* is ``None`` (unconstrained), returns an all-``True`` mask.  If *n*
    is given the mask has length *n*; otherwise it has length 0 for backward
    compatibility.

    *eps* is a feasibility tolerance: constraints with ``g(x) <= eps`` are
    treated as satisfied (default ``0.0``).
    """
    if G is None:
        return np.ones(n or 0, dtype=bool)
    return np.asarray(np.all(G <= eps, axis=1), dtype=bool)


__all__ = ["compute_violation", "is_feasible"]
