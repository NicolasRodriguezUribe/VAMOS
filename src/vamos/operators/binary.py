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
        def _bit_flip_masked(X: np.ndarray, mask: np.ndarray):
            rows, cols = mask.shape
            for i in range(rows):
                for j in range(cols):
                    if mask[i, j]:
                        X[i, j] = 1 - X[i, j]


def random_binary_population(pop_size: int, n_var: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a batch of random bitstrings.
    """
    if pop_size <= 0 or n_var <= 0:
        raise ValueError("pop_size and n_var must be positive integers.")
    return rng.integers(0, 2, size=(pop_size, n_var), dtype=np.int8)


def _as_pairs(X_parents: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Validate parent array and reshape into mating pairs.
    Accepts either (n_parents, n_var) or already-paired (n_pairs, 2, n_var).
    """
    if X_parents.ndim == 3 and X_parents.shape[1] == 2:
        pairs = X_parents
        D = X_parents.shape[2]
        return pairs, D

    Np, D = X_parents.shape
    if Np == 0:
        return np.empty((0, 2, D), dtype=X_parents.dtype), D
    # Handle odd parent count by duplicating the last parent
    if Np % 2 != 0:
        X_parents = np.vstack([X_parents, X_parents[-1:]])
        Np += 1
    return X_parents.reshape(Np // 2, 2, D).copy(), D


def one_point_crossover(X_parents: np.ndarray, prob: float, rng: np.random.Generator) -> np.ndarray:
    """
    Classic one-point crossover for bitstrings.
    """
    pairs, D = _as_pairs(X_parents)
    if pairs.size == 0 or D < 2:
        return pairs.reshape(X_parents.shape)
    prob = float(np.clip(prob, 0.0, 1.0))
    if prob <= 0.0:
        return pairs.reshape(X_parents.shape)

    active = rng.random(pairs.shape[0]) <= prob
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return pairs.reshape(X_parents.shape)

    cuts = rng.integers(1, D, size=idx.size)
    for row, cut in zip(idx, cuts):
        p1, p2 = pairs[row, 0], pairs[row, 1]
        child1 = np.concatenate([p1[:cut], p2[cut:]])
        child2 = np.concatenate([p2[:cut], p1[cut:]])
        pairs[row, 0], pairs[row, 1] = child1, child2
    return pairs.reshape(X_parents.shape)


def two_point_crossover(X_parents: np.ndarray, prob: float, rng: np.random.Generator) -> np.ndarray:
    """
    Two-point crossover for bitstrings.
    """
    pairs, D = _as_pairs(X_parents)
    if pairs.size == 0 or D < 2:
        return pairs.reshape(X_parents.shape)
    prob = float(np.clip(prob, 0.0, 1.0))
    if prob <= 0.0:
        return pairs.reshape(X_parents.shape)

    active = rng.random(pairs.shape[0]) <= prob
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return pairs.reshape(X_parents.shape)

    cuts = rng.integers(0, D, size=(idx.size, 2))
    lo = np.minimum(cuts[:, 0], cuts[:, 1])
    hi = np.maximum(cuts[:, 0], cuts[:, 1])
    hi = np.maximum(hi, lo + 1)
    hi = np.minimum(hi, D)

    for row, start, end in zip(idx, lo, hi):
        p1, p2 = pairs[row, 0], pairs[row, 1]
        child1 = p1.copy()
        child2 = p2.copy()
        child1[start:end] = p2[start:end]
        child2[start:end] = p1[start:end]
        pairs[row, 0], pairs[row, 1] = child1, child2
    return pairs.reshape(X_parents.shape)


def uniform_crossover(X_parents: np.ndarray, prob: float, rng: np.random.Generator) -> np.ndarray:
    """
    Uniform crossover with independent swapping per gene.
    """
    pairs, D = _as_pairs(X_parents)
    if pairs.size == 0:
        return pairs.reshape(X_parents.shape)
    prob = float(np.clip(prob, 0.0, 1.0))
    if prob <= 0.0:
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


def hux_crossover(X_parents: np.ndarray, prob: float, rng: np.random.Generator) -> np.ndarray:
    """
    Half-uniform crossover (HUX) for bitstrings.
    Swaps half of the differing bits between each parent pair.
    """
    pairs, D = _as_pairs(X_parents)
    if pairs.size == 0:
        return pairs.reshape(X_parents.shape)
    prob = float(np.clip(prob, 0.0, 1.0))
    if prob <= 0.0:
        return pairs.reshape(X_parents.shape)

    active = rng.random(pairs.shape[0]) <= prob
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return pairs.reshape(X_parents.shape)

    for row in idx:
        p1, p2 = pairs[row, 0], pairs[row, 1]
        diff_mask = p1 != p2
        if not np.any(diff_mask):
            continue
        diff_idx = np.flatnonzero(diff_mask)
        swap_count = max(1, diff_idx.size // 2)
        chosen = rng.choice(diff_idx, size=swap_count, replace=False)
        child1 = p1.copy()
        child2 = p2.copy()
        child1[chosen] = p2[chosen]
        child2[chosen] = p1[chosen]
        pairs[row, 0], pairs[row, 1] = child1, child2
    return pairs.reshape(X_parents.shape)


def bit_flip_mutation(X: np.ndarray, prob: float, rng: np.random.Generator) -> None:
    """
    Per-bit mutation that flips each bit with probability `prob`.
    """
    if X.size == 0:
        return
    prob = float(np.clip(prob, 0.0, 1.0))
    if prob <= 0.0:
        return
    mask = rng.random(X.shape) <= prob
    if _HAS_NUMBA:
        _bit_flip_masked(X, mask)
    else:
        X[mask] = 1 - X[mask]


__all__ = [
    "random_binary_population",
    "one_point_crossover",
    "two_point_crossover",
    "uniform_crossover",
    "hux_crossover",
    "bit_flip_mutation",
]
