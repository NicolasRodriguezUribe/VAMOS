from __future__ import annotations

from typing import Any, Callable, TypeAlias

import numpy as np
from numpy.typing import NDArray

PermArray: TypeAlias = NDArray[np.integer[Any]]
PermVec: TypeAlias = PermArray
PermPop: TypeAlias = PermArray
IndexArray: TypeAlias = NDArray[np.integer[Any]]
RNG: TypeAlias = np.random.Generator
CrossoverBuilder: TypeAlias = Callable[[PermVec, PermVec, RNG], tuple[PermVec, PermVec]]
RowMutation: TypeAlias = Callable[[PermVec, RNG], None]
Adjacency: TypeAlias = list[set[int]]

_SWAP_ROWS_JIT = None


def _use_numba_variation() -> bool:
    import os

    return os.environ.get("VAMOS_USE_NUMBA_VARIATION", "").lower() in {"1", "true", "yes"}


def _get_swap_rows_jit():
    global _SWAP_ROWS_JIT
    if _SWAP_ROWS_JIT is False:
        return None
    if _SWAP_ROWS_JIT is not None:
        return _SWAP_ROWS_JIT
    if not _use_numba_variation():
        _SWAP_ROWS_JIT = False
        return None
    try:
        from numba import njit
    except ImportError:
        _SWAP_ROWS_JIT = False
        return None

    @njit(cache=True)
    def _swap_rows_jit(X: np.ndarray, rows: np.ndarray, first: np.ndarray, second: np.ndarray) -> None:
        for idx in range(rows.shape[0]):
            r = rows[idx]
            a = first[idx]
            b = second[idx]
            tmp = X[r, a]
            X[r, a] = X[r, b]
            X[r, b] = tmp

    _SWAP_ROWS_JIT = _swap_rows_jit
    return _SWAP_ROWS_JIT


def random_permutation_population(
    pop_size: int,
    n_var: int,
    rng: RNG,
) -> PermPop:
    """
    Generate a batch of random permutations using the random-keys method.
    """
    if pop_size <= 0 or n_var <= 0:
        raise ValueError("pop_size and n_var must be positive integers.")
    keys = rng.random((pop_size, n_var))
    return np.argsort(keys, axis=1).astype(np.int32, copy=False)


def _ensure_distinct_indices(idx: IndexArray, upper: int, rng: RNG) -> None:
    if idx.size == 0:
        return
    same = idx[:, 0] == idx[:, 1]
    while np.any(same):
        idx[same, 1] = rng.integers(0, upper, size=int(np.count_nonzero(same)))
        same = idx[:, 0] == idx[:, 1]


def swap_mutation(
    X: PermPop,
    prob: float,
    rng: RNG,
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
    jit_fn = _get_swap_rows_jit()
    if jit_fn is not None:
        jit_fn(X, rows.astype(np.int64), first.astype(np.int64), second.astype(np.int64))
    else:
        tmp = X[rows, first].copy()
        X[rows, first] = X[rows, second]
        X[rows, second] = tmp


def pmx_crossover(
    X_parents: PermPop,
    prob: float,
    rng: RNG,
) -> PermPop:
    """
    Partially matched crossover (PMX) on permutation pairs.
    """
    return _pairwise_crossover(X_parents, prob, rng, _pmx_children)


def cycle_crossover(
    X_parents: PermPop,
    prob: float,
    rng: RNG,
) -> PermPop:
    """
    Cycle crossover (CX) on permutation pairs.
    """
    return _pairwise_crossover(X_parents, prob, rng, _cycle_children)


def position_based_crossover(
    X_parents: PermPop,
    prob: float,
    rng: RNG,
) -> PermPop:
    """
    Position-based crossover (POS). A random subset of positions from parent A is
    kept; remaining slots are filled with genes from parent B preserving order.
    """
    return _pairwise_crossover(X_parents, prob, rng, _position_based_children)


def edge_recombination_crossover(
    X_parents: PermPop,
    prob: float,
    rng: RNG,
) -> PermPop:
    """
    Edge recombination crossover (ERX) preserving adjacency information.
    """
    return _pairwise_crossover(X_parents, prob, rng, _edge_recombination_children)


def order_crossover(
    X_parents: PermPop,
    prob: float,
    rng: RNG,
) -> PermPop:
    """
    Batched order crossover (OX) for permutation-encoded chromosomes.
    """
    Np, D = X_parents.shape
    if Np == 0:
        return np.empty_like(X_parents)
    # Handle odd parent count by duplicating the last parent
    if Np % 2 != 0:
        X_parents = np.vstack([X_parents, X_parents[-1:]])
        Np += 1
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
        lo = int(cut_low[row])
        hi = int(cut_high[row])
        lo, hi = _ensure_valid_segment(length=D, lo=lo, hi=hi)

        p1, p2 = parents[pair_idx, 0], parents[pair_idx, 1]
        child1 = parents[pair_idx, 0].copy()
        child2 = parents[pair_idx, 1].copy()
        _order_crossover_into(p1, p2, child1, lo, hi)
        _order_crossover_into(p2, p1, child2, lo, hi)
        parents[pair_idx, 0], parents[pair_idx, 1] = child1, child2

    return parents.reshape(Np, D)


def insert_mutation(
    X: PermPop,
    prob: float,
    rng: RNG,
) -> None:
    """
    Remove one gene and insert it at another position (shift in-between genes).
    """
    _apply_row_mutation(X, prob, rng, _insert_row_mutation)


def scramble_mutation(
    X: PermPop,
    prob: float,
    rng: RNG,
) -> None:
    """
    Shuffle the genes inside a randomly chosen segment.
    """
    _apply_row_mutation(X, prob, rng, _scramble_row_mutation)


def inversion_mutation(
    X: PermPop,
    prob: float,
    rng: RNG,
) -> None:
    """
    Reverse a randomly chosen segment (standard inversion).
    """
    _apply_row_mutation(X, prob, rng, _inversion_row_mutation)


def simple_inversion_mutation(
    X: PermPop,
    prob: float,
    rng: RNG,
) -> None:
    """
    Alias of inversion_mutation for compatibility with jMetal naming.
    """
    inversion_mutation(X, prob, rng)


def displacement_mutation(
    X: PermPop,
    prob: float,
    rng: RNG,
) -> None:
    """
    Remove a contiguous segment and reinsert it at a different position.
    """
    _apply_row_mutation(X, prob, rng, _displacement_row_mutation)


# === Internal helpers ===


def _ensure_valid_segment(length: int, lo: int, hi: int) -> tuple[int, int]:
    if length < 2:
        return 0, 0
    if hi <= lo:
        hi = lo + 1
    if hi > length:
        hi = length
    return lo, hi


def _order_crossover_into(
    donor: PermVec,
    filler: PermVec,
    out: PermVec,
    cut1: int,
    cut2: int,
) -> None:
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


def _pairwise_crossover(
    X_parents: PermPop,
    prob: float,
    rng: RNG,
    builder: CrossoverBuilder,
) -> PermPop:
    Np, D = X_parents.shape
    if Np == 0:
        return np.empty_like(X_parents)
    # Handle odd parent count by duplicating the last parent
    if Np % 2 != 0:
        X_parents = np.vstack([X_parents, X_parents[-1:]])
        Np += 1
    prob = float(np.clip(prob, 0.0, 1.0))
    pairs = X_parents.reshape(Np // 2, 2, D).copy()
    if prob <= 0.0:
        return pairs.reshape(Np, D)

    mask = rng.random(pairs.shape[0]) <= prob
    active_idx = np.flatnonzero(mask)
    for pair_idx in active_idx:
        p1, p2 = pairs[pair_idx, 0], pairs[pair_idx, 1]
        c1, c2 = builder(p1, p2, rng)
        pairs[pair_idx, 0], pairs[pair_idx, 1] = c1, c2
    return pairs.reshape(Np, D)


def _pmx_children(p1: PermVec, p2: PermVec, rng: RNG) -> tuple[PermVec, PermVec]:
    n = p1.size
    c1, c2 = p1.copy(), p2.copy()
    cut1, cut2 = _two_cut_points(n, rng)
    _pmx_into(p1, p2, c1, cut1, cut2)
    _pmx_into(p2, p1, c2, cut1, cut2)
    return c1, c2


def _pmx_into(parent_a: PermVec, parent_b: PermVec, child: PermVec, cut1: int, cut2: int) -> None:
    n = parent_a.size
    cut1, cut2 = _ensure_valid_segment(n, cut1, cut2)
    child.fill(-1)
    used = np.zeros(n, dtype=bool)
    mapping = {}

    for i in range(cut1, cut2):
        gene_a = parent_a[i]
        gene_b = parent_b[i]
        child[i] = gene_a
        used[gene_a] = True
        mapping[gene_b] = gene_a

    for idx in np.concatenate([np.arange(0, cut1), np.arange(cut2, n)]):
        gene = parent_b[idx]
        seen = set()
        while gene in mapping and gene not in seen:
            seen.add(gene)
            gene = mapping[gene]
        if used[gene]:
            # Fallback to the next unused gene in parent_b scan order
            for candidate in parent_b:
                if not used[candidate]:
                    gene = candidate
                    break
        child[idx] = gene
        used[gene] = True


def _cycle_children(p1: PermVec, p2: PermVec, rng: RNG) -> tuple[PermVec, PermVec]:
    n = p1.size
    c1 = np.empty_like(p1)
    c2 = np.empty_like(p2)
    pos_in_p2 = np.empty(n, dtype=int)
    pos_in_p2[p2] = np.arange(n, dtype=int)

    used = np.zeros(n, dtype=bool)
    cycle_idx = 0
    for start in range(n):
        if used[start]:
            continue
        idx = start
        current_cycle = []
        while not used[idx]:
            used[idx] = True
            current_cycle.append(idx)
            idx = pos_in_p2[p1[idx]]
        pick_from_p1 = (cycle_idx % 2) == 0
        if pick_from_p1:
            c1[current_cycle] = p1[current_cycle]
            c2[current_cycle] = p2[current_cycle]
        else:
            c1[current_cycle] = p2[current_cycle]
            c2[current_cycle] = p1[current_cycle]
        cycle_idx += 1
    return c1, c2


def _position_based_children(p1: PermVec, p2: PermVec, rng: RNG) -> tuple[PermVec, PermVec]:
    n = p1.size
    c1 = np.full(n, -1, dtype=p1.dtype)
    c2 = np.full(n, -1, dtype=p2.dtype)

    k = rng.integers(1, n + 1)
    positions = rng.choice(n, size=k, replace=False)
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

    c1 = _edge_recombination_single(adj, p1, p2, rng)
    c2 = _edge_recombination_single(adj, p2, p1, rng)
    return c1, c2


def _edge_recombination_single(
    adj_template: Adjacency,
    parent_a: PermVec,
    parent_b: PermVec,
    rng: RNG,
) -> PermVec:
    n = parent_a.size
    adj = [set(neigh) for neigh in adj_template]
    child: PermVec = np.empty(n, dtype=parent_a.dtype)
    used = np.zeros(n, dtype=bool)

    current = parent_a[0] if rng.random() < 0.5 else parent_b[0]
    for pos in range(n):
        child[pos] = current
        used[current] = True
        # Remove current from all adjacency lists
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


def _apply_row_mutation(
    X: PermPop,
    prob: float,
    rng: RNG,
    mut_fn: RowMutation,
) -> None:
    N, D = X.shape
    if N == 0 or D < 2:
        return
    prob = float(np.clip(prob, 0.0, 1.0))
    if prob <= 0.0:
        return
    rows = np.flatnonzero(rng.random(N) <= prob)
    for idx in rows:
        mut_fn(X[idx], rng)


def _two_cut_points(length: int, rng: RNG) -> tuple[int, int]:
    if length < 2:
        return 0, 0
    a = rng.integers(0, length)
    b = rng.integers(0, length - 1)
    if b >= a:
        b += 1
    if a > b:
        a, b = b, a
    return int(a), int(b)


def _insert_row_mutation(row: PermVec, rng: RNG) -> None:
    n = row.size
    i, j = _two_cut_points(n, rng)
    gene = row[i]
    if i < j:
        row[i:j] = row[i + 1 : j + 1]
        row[j] = gene
    else:
        row[j + 1 : i + 1] = row[j:i]
        row[j] = gene


def _scramble_row_mutation(row: PermVec, rng: RNG) -> None:
    n = row.size
    lo, hi = _two_cut_points(n, rng)
    segment = row[lo:hi].copy()
    rng.shuffle(segment)
    row[lo:hi] = segment


def _inversion_row_mutation(row: PermVec, rng: RNG) -> None:
    n = row.size
    lo, hi = _two_cut_points(n, rng)
    row[lo:hi] = row[lo:hi][::-1]


def _displacement_row_mutation(row: PermVec, rng: RNG) -> None:
    n = row.size
    lo, hi = _two_cut_points(n, rng)
    segment = row[lo:hi].copy()
    remaining = np.concatenate([row[:lo], row[hi:]])
    insert_pos = rng.integers(0, remaining.size + 1)
    row[:] = np.concatenate([remaining[:insert_pos], segment, remaining[insert_pos:]])


# === Adapters ===

class SwapMutation:
    def __init__(self, prob: float = 0.1, **kwargs):
        self.prob = float(prob)
    def __call__(self, X: PermPop, rng: RNG, **kwargs) -> None:
        swap_mutation(X, self.prob, rng)

class PMXCrossover:
    def __init__(self, prob: float = 0.9, **kwargs):
        self.prob = float(prob)
    def __call__(self, parents: PermPop, rng: RNG, **kwargs) -> PermPop:
        return pmx_crossover(parents, self.prob, rng)

class CycleCrossover:
    def __init__(self, prob: float = 0.9, **kwargs):
        self.prob = float(prob)
    def __call__(self, parents: PermPop, rng: RNG, **kwargs) -> PermPop:
        return cycle_crossover(parents, self.prob, rng)

class PositionBasedCrossover:
    def __init__(self, prob: float = 0.9, **kwargs):
        self.prob = float(prob)
    def __call__(self, parents: PermPop, rng: RNG, **kwargs) -> PermPop:
        return position_based_crossover(parents, self.prob, rng)

class EdgeRecombinationCrossover:
    def __init__(self, prob: float = 0.9, **kwargs):
        self.prob = float(prob)
    def __call__(self, parents: PermPop, rng: RNG, **kwargs) -> PermPop:
        return edge_recombination_crossover(parents, self.prob, rng)

class OrderCrossover:
    def __init__(self, prob: float = 0.9, **kwargs):
        self.prob = float(prob)
    def __call__(self, parents: PermPop, rng: RNG, **kwargs) -> PermPop:
        return order_crossover(parents, self.prob, rng)

class InsertMutation:
    def __init__(self, prob: float = 0.1, **kwargs):
        self.prob = float(prob)
    def __call__(self, X: PermPop, rng: RNG, **kwargs) -> None:
        insert_mutation(X, self.prob, rng)

class ScrambleMutation:
    def __init__(self, prob: float = 0.1, **kwargs):
        self.prob = float(prob)
    def __call__(self, X: PermPop, rng: RNG, **kwargs) -> None:
        scramble_mutation(X, self.prob, rng)

class InversionMutation:
    def __init__(self, prob: float = 0.1, **kwargs):
        self.prob = float(prob)
    def __call__(self, X: PermPop, rng: RNG, **kwargs) -> None:
        inversion_mutation(X, self.prob, rng)

class SimpleInversionMutation:
    def __init__(self, prob: float = 0.1, **kwargs):
        self.prob = float(prob)
    def __call__(self, X: PermPop, rng: RNG, **kwargs) -> None:
        simple_inversion_mutation(X, self.prob, rng)

class DisplacementMutation:
    def __init__(self, prob: float = 0.1, **kwargs):
        self.prob = float(prob)
    def __call__(self, X: PermPop, rng: RNG, **kwargs) -> None:
        displacement_mutation(X, self.prob, rng)

__all__ = [
    "random_permutation_population",
    "swap_mutation",
    "pmx_crossover",
    "cycle_crossover",
    "position_based_crossover",
    "edge_recombination_crossover",
    "order_crossover",
    "insert_mutation",
    "scramble_mutation",
    "inversion_mutation",
    "simple_inversion_mutation",
    "displacement_mutation",
    "SwapMutation",
    "PMXCrossover",
    "CycleCrossover",
    "PositionBasedCrossover",
    "EdgeRecombinationCrossover",
    "OrderCrossover",
    "InsertMutation",
    "ScrambleMutation",
    "InversionMutation",
    "SimpleInversionMutation",
    "DisplacementMutation",
]
