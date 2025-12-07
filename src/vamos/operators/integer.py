from __future__ import annotations

import os
import numpy as np

_USE_NUMBA_VARIATION = os.environ.get("VAMOS_USE_NUMBA_VARIATION", "").lower() in {"1", "true", "yes"}
_HAS_NUMBA = False
if _USE_NUMBA_VARIATION:
    try:
        from numba import njit
    except ImportError:
        _HAS_NUMBA = False
    else:
        _HAS_NUMBA = True

        @njit(cache=True)
        def _random_reset_masked(X: np.ndarray, mask: np.ndarray, rand_vals: np.ndarray):
            rows, cols = mask.shape
            for i in range(rows):
                for j in range(cols):
                    if mask[i, j]:
                        X[i, j] = rand_vals[i, j]


def random_integer_population(
    pop_size: int, n_var: int, lower: np.ndarray, upper: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """
    Generate integer-valued individuals within inclusive [lower, upper].
    """
    if pop_size <= 0 or n_var <= 0:
        raise ValueError("pop_size and n_var must be positive integers.")
    if lower.shape != (n_var,) or upper.shape != (n_var,):
        raise ValueError("lower/upper must be 1D arrays matching n_var.")
    return rng.integers(lower, upper + 1, size=(pop_size, n_var), dtype=np.int32)


def _as_pairs(X_parents: np.ndarray) -> tuple[np.ndarray, int]:
    if X_parents.ndim == 3 and X_parents.shape[1] == 2:
        return X_parents, X_parents.shape[2]
    Np, D = X_parents.shape
    if Np == 0:
        return np.empty((0, 2, D), dtype=X_parents.dtype), D
    # Handle odd parent count by duplicating the last parent
    if Np % 2 != 0:
        X_parents = np.vstack([X_parents, X_parents[-1:]])
        Np += 1
    return X_parents.reshape(Np // 2, 2, D).copy(), D


def uniform_integer_crossover(
    X_parents: np.ndarray, prob: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Per-gene uniform crossover for integer vectors.
    """
    pairs, D = _as_pairs(X_parents)
    prob = float(np.clip(prob, 0.0, 1.0))
    if pairs.size == 0 or prob <= 0.0:
        return pairs.reshape(X_parents.shape)
    active = rng.random(pairs.shape[0]) <= prob
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return pairs.reshape(X_parents.shape)
    swap_mask = rng.random((idx.size, D)) < 0.5
    for row, mask in zip(idx, swap_mask):
        p1, p2 = pairs[row, 0], pairs[row, 1]
        child1 = np.where(mask, p1, p2)
        child2 = np.where(mask, p2, p1)
        pairs[row, 0], pairs[row, 1] = child1, child2
    return pairs.reshape(X_parents.shape)


def arithmetic_integer_crossover(
    X_parents: np.ndarray, prob: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Integer arithmetic crossover: average parents and round.
    """
    pairs, _ = _as_pairs(X_parents)
    prob = float(np.clip(prob, 0.0, 1.0))
    if pairs.size == 0 or prob <= 0.0:
        return pairs.reshape(X_parents.shape)
    active = rng.random(pairs.shape[0]) <= prob
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return pairs.reshape(X_parents.shape)
    for row in idx:
        p1, p2 = pairs[row, 0], pairs[row, 1]
        mean = np.rint(0.5 * (p1 + p2)).astype(p1.dtype, copy=False)
        pairs[row, 0] = mean
        pairs[row, 1] = mean
    return pairs.reshape(X_parents.shape)


def random_reset_mutation(
    X: np.ndarray, prob: float, lower: np.ndarray, upper: np.ndarray, rng: np.random.Generator
) -> None:
    """
    Per-gene random reset to any value within bounds.
    """
    if X.size == 0:
        return
    prob = float(np.clip(prob, 0.0, 1.0))
    if prob <= 0.0:
        return
    if lower.shape != upper.shape or lower.shape[0] != X.shape[1]:
        raise ValueError("lower/upper must match chromosome length.")
    mask = rng.random(X.shape) <= prob
    if not np.any(mask):
        return
    rand_vals = rng.integers(lower, upper + 1, size=X.shape, dtype=X.dtype)
    if _HAS_NUMBA:
        _random_reset_masked(X, mask, rand_vals)
    else:
        X[mask] = rand_vals[mask]


def creep_mutation(
    X: np.ndarray, prob: float, step: int, lower: np.ndarray, upper: np.ndarray, rng: np.random.Generator
) -> None:
    """
    Small integer step mutation (+/- step).
    """
    if X.size == 0:
        return
    prob = float(np.clip(prob, 0.0, 1.0))
    if prob <= 0.0:
        return
    if lower.shape != upper.shape or lower.shape[0] != X.shape[1]:
        raise ValueError("lower/upper must match chromosome length.")
    mask = rng.random(X.shape) <= prob
    if not np.any(mask):
        return
    deltas = rng.choice([-step, step], size=X.shape, replace=True)
    proposed = X.copy()
    proposed[mask] = proposed[mask] + deltas[mask]
    np.clip(proposed, lower, upper, out=proposed)
    X[:] = proposed


__all__ = [
    "random_integer_population",
    "uniform_integer_crossover",
    "arithmetic_integer_crossover",
    "random_reset_mutation",
    "creep_mutation",
]
