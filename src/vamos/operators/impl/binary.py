from __future__ import annotations

from typing import Any
from collections.abc import Callable

import numpy as np

_BIT_FLIP_MASKED_JIT: Callable[[np.ndarray, np.ndarray], None] | None = None
_BIT_FLIP_MASKED_DISABLED = False


def _use_numba_variation() -> bool:
    import os

    return os.environ.get("VAMOS_USE_NUMBA_VARIATION", "").lower() in {"1", "true", "yes"}


def _get_bit_flip_masked() -> Callable[[np.ndarray, np.ndarray], None] | None:
    global _BIT_FLIP_MASKED_JIT, _BIT_FLIP_MASKED_DISABLED
    if _BIT_FLIP_MASKED_DISABLED:
        return None
    if _BIT_FLIP_MASKED_JIT is not None:
        return _BIT_FLIP_MASKED_JIT
    if not _use_numba_variation():
        _BIT_FLIP_MASKED_DISABLED = True
        return None
    try:
        from numba import njit
    except ImportError:
        _BIT_FLIP_MASKED_DISABLED = True
        return None

    @njit(cache=True)  # type: ignore[untyped-decorator]
    def _bit_flip_masked(X: np.ndarray, mask: np.ndarray) -> None:
        rows, cols = mask.shape
        for i in range(rows):
            for j in range(cols):
                if mask[i, j]:
                    X[i, j] = 1 - X[i, j]

    _BIT_FLIP_MASKED_JIT = _bit_flip_masked
    return _BIT_FLIP_MASKED_JIT


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


def _reshape_offspring(pairs: np.ndarray, parents: np.ndarray) -> np.ndarray:
    if parents.ndim == 2 and parents.shape[0] % 2 != 0:
        return pairs.reshape(-1, pairs.shape[2])[: parents.shape[0]]
    return pairs.reshape(parents.shape)


def one_point_crossover(X_parents: np.ndarray, prob: float, rng: np.random.Generator) -> np.ndarray:
    """
    Classic one-point crossover for bitstrings.
    """
    pairs, D = _as_pairs(X_parents)
    if pairs.size == 0 or D < 2:
        return _reshape_offspring(pairs, X_parents)
    prob = float(np.clip(prob, 0.0, 1.0))
    if prob <= 0.0:
        return _reshape_offspring(pairs, X_parents)

    active = rng.random(pairs.shape[0]) <= prob
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return _reshape_offspring(pairs, X_parents)

    cuts = rng.integers(1, D, size=idx.size)
    for row, cut in zip(idx, cuts):
        p1, p2 = pairs[row, 0], pairs[row, 1]
        child1 = np.concatenate([p1[:cut], p2[cut:]])
        child2 = np.concatenate([p2[:cut], p1[cut:]])
        pairs[row, 0], pairs[row, 1] = child1, child2
    return _reshape_offspring(pairs, X_parents)


def two_point_crossover(X_parents: np.ndarray, prob: float, rng: np.random.Generator) -> np.ndarray:
    """
    Two-point crossover for bitstrings.
    """
    pairs, D = _as_pairs(X_parents)
    if pairs.size == 0 or D < 2:
        return _reshape_offspring(pairs, X_parents)
    prob = float(np.clip(prob, 0.0, 1.0))
    if prob <= 0.0:
        return _reshape_offspring(pairs, X_parents)

    active = rng.random(pairs.shape[0]) <= prob
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return _reshape_offspring(pairs, X_parents)

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
    return _reshape_offspring(pairs, X_parents)


def uniform_crossover(X_parents: np.ndarray, prob: float, rng: np.random.Generator) -> np.ndarray:
    """
    Uniform crossover with independent swapping per gene.
    """
    pairs, D = _as_pairs(X_parents)
    if pairs.size == 0:
        return _reshape_offspring(pairs, X_parents)
    prob = float(np.clip(prob, 0.0, 1.0))
    if prob <= 0.0:
        return _reshape_offspring(pairs, X_parents)

    active = rng.random(pairs.shape[0]) <= prob
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return _reshape_offspring(pairs, X_parents)

    swap_mask = rng.random((idx.size, D)) < 0.5
    for row, mask in zip(idx, swap_mask):
        p1, p2 = pairs[row, 0], pairs[row, 1]
        child1 = np.where(mask, p1, p2)
        child2 = np.where(mask, p2, p1)
        pairs[row, 0], pairs[row, 1] = child1, child2
    return _reshape_offspring(pairs, X_parents)


def hux_crossover(X_parents: np.ndarray, prob: float, rng: np.random.Generator) -> np.ndarray:
    """
    Half-uniform crossover (HUX) for bitstrings.
    Swaps half of the differing bits between each parent pair.
    """
    pairs, D = _as_pairs(X_parents)
    if pairs.size == 0:
        return _reshape_offspring(pairs, X_parents)
    prob = float(np.clip(prob, 0.0, 1.0))
    if prob <= 0.0:
        return _reshape_offspring(pairs, X_parents)

    active = rng.random(pairs.shape[0]) <= prob
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return _reshape_offspring(pairs, X_parents)

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
    return _reshape_offspring(pairs, X_parents)


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
    jit_fn = _get_bit_flip_masked()
    if jit_fn is not None:
        jit_fn(X, mask)
    else:
        X[mask] = 1 - X[mask]


class BitFlipMutation:
    def __init__(self, prob: float = 0.1, **kwargs: Any) -> None:
        self.prob = float(prob)

    def __call__(self, X: np.ndarray, rng: np.random.Generator, **kwargs: Any) -> np.ndarray:
        # Mutation in place
        bit_flip_mutation(X, self.prob, rng)
        return X


class OnePointCrossover:
    def __init__(self, prob: float = 0.9, **kwargs: Any) -> None:
        self.prob = float(prob)

    def __call__(self, parents: np.ndarray, rng: np.random.Generator, **kwargs: Any) -> np.ndarray:
        return one_point_crossover(parents, self.prob, rng)


class TwoPointCrossover:
    def __init__(self, prob: float = 0.9, **kwargs: Any) -> None:
        self.prob = float(prob)

    def __call__(self, parents: np.ndarray, rng: np.random.Generator, **kwargs: Any) -> np.ndarray:
        return two_point_crossover(parents, self.prob, rng)


class UniformCrossover:
    def __init__(self, prob: float = 0.9, **kwargs: Any) -> None:
        self.prob = float(prob)

    def __call__(self, parents: np.ndarray, rng: np.random.Generator, **kwargs: Any) -> np.ndarray:
        return uniform_crossover(parents, self.prob, rng)


class HuxCrossover:
    def __init__(self, prob: float = 0.9, **kwargs: Any) -> None:
        self.prob = float(prob)

    def __call__(self, parents: np.ndarray, rng: np.random.Generator, **kwargs: Any) -> np.ndarray:
        return hux_crossover(parents, self.prob, rng)


__all__ = [
    "random_binary_population",
    "one_point_crossover",
    "two_point_crossover",
    "uniform_crossover",
    "hux_crossover",
    "bit_flip_mutation",
    "BitFlipMutation",
    "OnePointCrossover",
    "TwoPointCrossover",
    "UniformCrossover",
    "HuxCrossover",
]
