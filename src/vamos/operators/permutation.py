from __future__ import annotations

import numpy as np


def random_permutation_population(
    pop_size: int,
    n_var: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a batch of random permutations using the random-keys method.
    """
    if pop_size <= 0 or n_var <= 0:
        raise ValueError("pop_size and n_var must be positive integers.")
    keys = rng.random((pop_size, n_var))
    return np.argsort(keys, axis=1).astype(np.int32, copy=False)


def _ensure_distinct_indices(idx: np.ndarray, upper: int, rng: np.random.Generator) -> None:
    """
    Ensure each row in idx (shape *,2) contains distinct positions by resampling the clashes.
    """
    if idx.size == 0:
        return
    same = idx[:, 0] == idx[:, 1]
    while np.any(same):
        idx[same, 1] = rng.integers(0, upper, size=int(np.count_nonzero(same)))
        same = idx[:, 0] == idx[:, 1]


def order_crossover(
    X_parents: np.ndarray,
    prob: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Batched order crossover (OX) for permutation-encoded chromosomes.
    """
    Np, D = X_parents.shape
    if Np == 0:
        return np.empty_like(X_parents)
    if Np % 2 != 0:
        raise ValueError("Permutation crossover expects an even number of parents.")
    prob = float(np.clip(prob, 0.0, 1.0))
    parents = X_parents.reshape(Np // 2, 2, D).copy()
    if prob <= 0.0:
        return parents.reshape(Np, D)

    n_pairs = parents.shape[0]
    mask = rng.random(n_pairs) <= prob
    active_idx = np.flatnonzero(mask)
    if active_idx.size == 0:
        return parents.reshape(Np, D)

    cuts = rng.integers(0, D, size=(active_idx.size, 2))
    _ensure_distinct_indices(cuts, D, rng)
    cut_low = np.minimum(cuts[:, 0], cuts[:, 1])
    cut_high = np.maximum(cuts[:, 0], cuts[:, 1])

    for row, pair_idx in enumerate(active_idx):
        lo = cut_low[row]
        hi = cut_high[row]
        if hi <= lo:
            hi = lo + 1
        p1 = parents[pair_idx, 0]
        p2 = parents[pair_idx, 1]
        child1 = parents[pair_idx, 0].copy()
        child2 = parents[pair_idx, 1].copy()
        _order_crossover_into(p1, p2, child1, lo, hi)
        _order_crossover_into(p2, p1, child2, lo, hi)
        parents[pair_idx, 0] = child1
        parents[pair_idx, 1] = child2

    return parents.reshape(Np, D)


def _order_crossover_into(
    donor: np.ndarray,
    filler: np.ndarray,
    out: np.ndarray,
    cut1: int,
    cut2: int,
) -> None:
    """
    Helper that writes the OX child into `out` using donor/filler parents.
    """
    cut1 = int(cut1)
    cut2 = int(cut2)
    if cut1 == cut2:
        cut2 = cut1 + 1
    n = donor.size
    cut2 = min(cut2, n)
    out.fill(-1)
    out[cut1:cut2] = donor[cut1:cut2]
    # Track used genes to filter filler parent.
    used = np.zeros(n, dtype=bool)
    rows = donor[cut1:cut2]
    if rows.size:
        used[rows] = True

    filtered = filler[~used[filler]]
    fill_positions = np.concatenate([np.arange(cut2, n), np.arange(0, cut1)])
    out[fill_positions] = filtered


def swap_mutation(
    X: np.ndarray,
    prob: float,
    rng: np.random.Generator,
) -> None:
    """
    Batched swap mutation that randomly swaps two positions in selected individuals.
    """
    N, D = X.shape
    if N == 0:
        return
    prob = float(np.clip(prob, 0.0, 1.0))
    if prob <= 0.0:
        return
    mutate_mask = rng.random(N) <= prob
    rows = np.flatnonzero(mutate_mask)
    if rows.size == 0:
        return
    if D < 2:
        return
    idx_pairs = rng.integers(0, D, size=(rows.size, 2))
    _ensure_distinct_indices(idx_pairs, D, rng)
    first = idx_pairs[:, 0]
    second = idx_pairs[:, 1]
    tmp = X[rows, first].copy()
    X[rows, first] = X[rows, second]
    X[rows, second] = tmp
