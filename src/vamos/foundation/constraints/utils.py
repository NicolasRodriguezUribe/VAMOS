"""
Utility helpers for constraint handling.
"""

from __future__ import annotations

import numpy as np


def compute_violation(G: np.ndarray) -> np.ndarray:
    """
    Sum of positive parts per-solution; assumes G shape (N, n_constr), g<=0 satisfied.
    """
    if G is None:
        return np.zeros(0, dtype=float)
    positive = np.maximum(G, 0.0)
    return np.sum(positive, axis=1)


def is_feasible(G: np.ndarray) -> np.ndarray:
    """
    Boolean feasibility mask; assumes G shape (N, n_constr).
    """
    if G is None:
        return np.array([], dtype=bool)
    return np.all(G <= 0.0, axis=1)


__all__ = ["compute_violation", "is_feasible"]
