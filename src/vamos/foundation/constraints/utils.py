"""
Utility helpers for constraint handling.
"""

from __future__ import annotations

import numpy as np


def compute_violation(G: np.ndarray | None, *, n: int | None = None) -> np.ndarray:
    """Sum of positive parts per-solution; assumes G shape (N, n_constraints), g<=0 satisfied.

    When *G* is ``None`` (unconstrained), ``n`` must be provided so the output
    length is explicit.
    """
    if G is None:
        if n is None:
            raise ValueError("compute_violation() requires 'n' when G is None.")
        return np.zeros(n, dtype=float)
    positive = np.maximum(G, 0.0)
    return np.asarray(np.sum(positive, axis=1), dtype=float)


def is_feasible(G: np.ndarray | None, *, n: int | None = None, eps: float = 0.0) -> np.ndarray:
    """Boolean feasibility mask; assumes G shape (N, n_constraints).

    When *G* is ``None`` (unconstrained), ``n`` must be provided so the output
    length is explicit.

    *eps* is a feasibility tolerance: constraints with ``g(x) <= eps`` are
    treated as satisfied (default ``0.0``).
    """
    if G is None:
        if n is None:
            raise ValueError("is_feasible() requires 'n' when G is None.")
        return np.ones(n, dtype=bool)
    return np.asarray(np.all(G <= eps, axis=1), dtype=bool)


__all__ = ["compute_violation", "is_feasible"]
