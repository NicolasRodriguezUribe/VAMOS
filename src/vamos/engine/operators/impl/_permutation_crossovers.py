from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._permutation_common import (
    RNG,
    Adjacency,
    CrossoverBuilder,
    PermPop,
    PermVec,
    ensure_distinct_indices,
    ensure_valid_segment,
    trim_offspring,
    two_cut_points,
    validate_permutation_population,
)


def pmx_crossover(X_parents: PermPop, prob: float, rng: RNG) -> PermPop:
    return _pairwise_crossover(X_parents, prob, rng, _pmx_children)


def cycle_crossover(X_parents: PermPop, prob: float, rng: RNG) -> PermPop:
    return _pairwise_crossover(X_parents, prob, rng, _cycle_children)


def position_based_crossover(X_parents: PermPop, prob: float, rng: RNG) -> PermPop:
    return _pairwise_crossover(X_parents, prob, rng, _position_based_children)


def edge_recombination_crossover(X_parents: PermPop, prob: float, rng: RNG) -> PermPop:
    return _pairwise_crossover(X_parents, prob, rng, _edge_recombination_children)


def order_crossover(X_parents: PermPop, prob: float, rng: RNG) -> PermPop:
    X_parents = validate_permutation_population(X_parents, label="X_parents")
    Np, D = X_parents.shape
    if Np == 0 or D < 2:
        return X_parents.copy()
    n_original = Np
    if Np % 2 != 0:
        X_parents = np.vstack([X_parents, X_parents[-1:]])
        Np += 1
    prob = float(np.clip(prob, 0.0, 1.0))
    parents = X_parents.reshape(Np // 2, 2, D).copy()
    if prob <= 0.0:
        return trim_offspring(parents.reshape(Np, D), n_original)

    active_idx = np.flatnonzero(rng.random(parents.shape[0]) <= prob)
    if active_idx.size == 0:
        return trim_offspring(parents.reshape(Np, D), n_original)

    cuts = rng.integers(0, D, size=(active_idx.size, 2))
    ensure_distinct_indices(cuts, D, rng)
    cut_low = np.minimum(cuts[:, 0], cuts[:, 1])
    cut_high = np.maximum(cuts[:, 0], cuts[:, 1])

    for row, pair_idx in enumerate(active_idx):
        lo, hi = ensure_valid_segment(length=D, lo=int(cut_low[row]), hi=int(cut_high[row]))
        p1, p2 = parents[pair_idx, 0], parents[pair_idx, 1]
        child1 = parents[pair_idx, 0].copy()
        child2 = parents[pair_idx, 1].copy()
        _order_crossover_into(p1, p2, child1, lo, hi)
        _order_crossover_into(p2, p1, child2, lo, hi)
        parents[pair_idx, 0], parents[pair_idx, 1] = child1, child2

    return trim_offspring(parents.reshape(Np, D), n_original)


def alternating_edges_crossover(X_parents: PermPop, prob: float, rng: RNG) -> PermPop:
    return _pairwise_crossover(X_parents, prob, rng, _alternating_edges_children)


def _order_crossover_into(donor: PermVec, filler: PermVec, out: PermVec, cut1: int, cut2: int) -> None:
    if cut1 == cut2:
        cut2 = cut1 + 1
    n = donor.size
    cut2 = min(cut2, n)
    out.fill(-1)
    out[cut1:cut2] = donor[cut1:cut2]
    used = np.zeros(n, dtype=bool)
    rows = donor[cut1:cut2]
    if rows.size:
        used[rows] = True

    filtered = filler[~used[filler]]
    fill_positions = np.concatenate([np.arange(cut2, n), np.arange(0, cut1)])
    out[fill_positions] = filtered


def _pairwise_crossover(
    X_parents: PermPop,
    prob: float,
    rng: RNG,
    builder: CrossoverBuilder,
) -> PermPop:
    X_parents = validate_permutation_population(X_parents, label="X_parents")
    Np, D = X_parents.shape
    if Np == 0:
        return np.empty_like(X_parents)
    n_original = Np
    if Np % 2 != 0:
        X_parents = np.vstack([X_parents, X_parents[-1:]])
        Np += 1
    prob = float(np.clip(prob, 0.0, 1.0))
    pairs = X_parents.reshape(Np // 2, 2, D).copy()
    if prob <= 0.0:
        return trim_offspring(pairs.reshape(Np, D), n_original)

    for pair_idx in np.flatnonzero(rng.random(pairs.shape[0]) <= prob):
        p1, p2 = pairs[pair_idx, 0], pairs[pair_idx, 1]
        c1, c2 = builder(p1, p2, rng)
        pairs[pair_idx, 0], pairs[pair_idx, 1] = c1, c2
    return trim_offspring(pairs.reshape(Np, D), n_original)


def _pmx_children(p1: PermVec, p2: PermVec, rng: RNG) -> tuple[PermVec, PermVec]:
    c1, c2 = p1.copy(), p2.copy()
    cut1, cut2 = two_cut_points(p1.size, rng)
    _pmx_into(p1, p2, c1, cut1, cut2)
    _pmx_into(p2, p1, c2, cut1, cut2)
    return c1, c2


def _pmx_into(parent_a: PermVec, parent_b: PermVec, child: PermVec, cut1: int, cut2: int) -> None:
    n = parent_a.size
    if n < 2:
        return
    if cut2 < cut1:
        cut1, cut2 = cut2, cut1
    cut1 = max(0, min(int(cut1), n - 1))
    cut2 = max(0, min(int(cut2), n - 1))

    mapping: dict[int, int] = {}
    for i in range(cut1, cut2 + 1):
        mapping[int(parent_b[i])] = int(parent_a[i])

    for i in range(n):
        if cut1 <= i <= cut2:
            child[i] = parent_b[i]
            continue
        gene = int(parent_a[i])
        steps = 0
        limit = len(mapping) + 1
        while gene in mapping and steps < limit:
            gene = mapping[gene]
            steps += 1
        child[i] = gene


def _cycle_children(p1: PermVec, p2: PermVec, rng: RNG) -> tuple[PermVec, PermVec]:
    n = p1.size
    if n == 0:
        return p1.copy(), p2.copy()
    c1 = p2.copy()
    c2 = p1.copy()
    pos_in_p1 = np.empty(n, dtype=int)
    pos_in_p1[p1] = np.arange(n, dtype=int)

    start_idx = int(rng.integers(0, n))
    cycle = []
    idx = start_idx
    while True:
        cycle.append(idx)
        idx = pos_in_p1[p2[idx]]
        if idx == start_idx:
            break

    cycle_idx = np.asarray(cycle, dtype=int)
    c1[cycle_idx] = p1[cycle_idx]
    c2[cycle_idx] = p2[cycle_idx]
    return c1, c2


def _position_based_children(p1: PermVec, p2: PermVec, rng: RNG) -> tuple[PermVec, PermVec]:
    n = p1.size
    c1 = np.full(n, -1, dtype=p1.dtype)
    c2 = np.full(n, -1, dtype=p2.dtype)
    positions = rng.choice(n, size=rng.integers(1, n + 1), replace=False)
    pos_mask = np.zeros(n, dtype=bool)
    pos_mask[positions] = True
    c1[pos_mask] = p1[pos_mask]
    c2[pos_mask] = p2[pos_mask]
    _fill_from_other_parent(c1, p2, pos_mask)
    _fill_from_other_parent(c2, p1, pos_mask)
    return c1, c2


def _fill_from_other_parent(child: PermVec, donor: PermVec, fixed_mask: NDArray[np.bool_]) -> None:
    n = donor.size
    used = np.zeros(n, dtype=bool)
    used[child[fixed_mask]] = True
    insert_positions = np.flatnonzero(~fixed_mask)
    idx = 0
    for gene in donor:
        if not used[gene]:
            child[insert_positions[idx]] = gene
            used[gene] = True
            idx += 1
            if idx == insert_positions.size:
                break


def _edge_recombination_children(p1: PermVec, p2: PermVec, rng: RNG) -> tuple[PermVec, PermVec]:
    n = p1.size
    adj: Adjacency = [set() for _ in range(n)]

    def add_edges(parent: PermVec) -> None:
        for i in range(n):
            gene = parent[i]
            left = parent[(i - 1) % n]
            right = parent[(i + 1) % n]
            adj[gene].add(left)
            adj[gene].add(right)

    add_edges(p1)
    add_edges(p2)
    return _edge_recombination_single(adj, p1, p2, rng), _edge_recombination_single(adj, p2, p1, rng)


def _edge_recombination_single(adj_template: Adjacency, parent_a: PermVec, parent_b: PermVec, rng: RNG) -> PermVec:
    n = parent_a.size
    adj = [set(neigh) for neigh in adj_template]
    child: PermVec = np.empty(n, dtype=parent_a.dtype)
    used = np.zeros(n, dtype=bool)
    current = parent_a[0] if rng.random() < 0.5 else parent_b[0]
    for pos in range(n):
        child[pos] = current
        used[current] = True
        for neighbors in adj:
            neighbors.discard(current)
        candidates = adj[current]
        if candidates:
            min_deg = min(len(adj[c]) for c in candidates)
            tight = [c for c in candidates if len(adj[c]) == min_deg]
            current = rng.choice(tight)
        else:
            remaining = np.flatnonzero(~used)
            if remaining.size == 0:
                break
            current = int(rng.choice(remaining))
    return child


def _alternating_edges_children(p1: PermVec, p2: PermVec, rng: RNG) -> tuple[PermVec, PermVec]:
    return _alternating_edges_child(p1, p2, rng, prefer_parent_a=True), _alternating_edges_child(p2, p1, rng, prefer_parent_a=False)


def _alternating_edges_child(parent_a: PermVec, parent_b: PermVec, rng: RNG, *, prefer_parent_a: bool) -> PermVec:
    n = parent_a.size
    child = np.empty(n, dtype=parent_a.dtype)
    if n == 0:
        return child

    pos_a = np.empty(n, dtype=int)
    pos_b = np.empty(n, dtype=int)
    pos_a[parent_a] = np.arange(n, dtype=int)
    pos_b[parent_b] = np.arange(n, dtype=int)
    used = np.zeros(n, dtype=bool)

    current = int(parent_a[0]) if prefer_parent_a else int(parent_b[0])
    take_from_a = prefer_parent_a
    child[0] = current
    used[current] = True

    for i in range(1, n):
        if take_from_a:
            primary = int(parent_a[(pos_a[current] + 1) % n])
            secondary = int(parent_b[(pos_b[current] + 1) % n])
        else:
            primary = int(parent_b[(pos_b[current] + 1) % n])
            secondary = int(parent_a[(pos_a[current] + 1) % n])

        if not used[primary]:
            nxt = primary
        elif not used[secondary]:
            nxt = secondary
        else:
            remaining = np.flatnonzero(~used)
            if remaining.size == 0:
                break
            nxt = int(rng.choice(remaining))

        child[i] = nxt
        used[nxt] = True
        current = nxt
        take_from_a = not take_from_a

    return child


__all__ = [
    "alternating_edges_crossover",
    "cycle_crossover",
    "edge_recombination_crossover",
    "order_crossover",
    "pmx_crossover",
    "position_based_crossover",
]
