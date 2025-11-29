from __future__ import annotations

import numpy as np


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
    """
    Np, D = X_parents.shape
    if Np == 0:
        return np.empty((0, 2, D), dtype=X_parents.dtype), D
    if Np % 2 != 0:
        raise ValueError("Binary crossover expects an even number of parents.")
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
    X[mask] = 1 - X[mask]


__all__ = [
    "random_binary_population",
    "one_point_crossover",
    "two_point_crossover",
    "uniform_crossover",
    "bit_flip_mutation",
]
