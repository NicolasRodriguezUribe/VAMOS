from __future__ import annotations

from typing import Literal, overload

import numpy as np


@overload
def pareto_filter(F: np.ndarray | None, *, return_indices: Literal[False] = False) -> np.ndarray | None: ...


@overload
def pareto_filter(F: np.ndarray | None, *, return_indices: Literal[True]) -> tuple[np.ndarray, np.ndarray]: ...


def pareto_filter(F: np.ndarray | None, *, return_indices: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray] | None:
    """
    Return the non-dominated subset of points (first Pareto front).

    Args:
        F: Objective values array (n_solutions, n_objectives) or None.
        return_indices: When True, also return indices of the front in F.

    Returns:
        Front array, or (front, indices) when return_indices is True.
    """
    if F is None:
        if return_indices:
            return np.empty((0, 0)), np.array([], dtype=int)
        return None
    F = np.asarray(F)
    if F.size == 0 or F.ndim < 2:
        if return_indices:
            n = int(F.shape[0]) if F.ndim > 0 else 0
            idx = np.arange(n, dtype=int)
            return F, idx
        return F
    from vamos.foundation.kernel.numpy_backend import _fast_non_dominated_sort

    fronts, _ = _fast_non_dominated_sort(F)
    if not fronts or not fronts[0]:
        if return_indices:
            idx = np.arange(int(F.shape[0]), dtype=int)
            return F, idx
        return F
    idx = np.asarray(fronts[0], dtype=int)
    front = F[idx]
    return (front, idx) if return_indices else front
