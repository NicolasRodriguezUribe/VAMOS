from __future__ import annotations

import numpy as np


def sum_violations_gte0(*g: np.ndarray) -> np.ndarray:
    """
    Return sum_i max{-g_i(x), 0} assuming each constraint is feasible when g_i(x) >= 0.
    """

    if not g:
        raise ValueError("At least one constraint array is required.")
    stacked = np.stack(g, axis=1)
    violations = np.maximum(-stacked, 0.0).sum(axis=1)
    return np.asarray(violations, dtype=float)


__all__ = ["sum_violations_gte0"]
