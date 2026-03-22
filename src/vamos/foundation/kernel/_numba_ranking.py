from __future__ import annotations

import numpy as np

from .numba_ops import njit


@njit(cache=True)
def _fast_non_dominated_sort_ranks_2d_sorted(F_sorted: np.ndarray, y_codes: np.ndarray) -> np.ndarray:
    N = F_sorted.shape[0]
    if N == 0:
        return np.empty(0, dtype=np.int64)

    bit = np.zeros(int(y_codes.max()) + 2, dtype=np.int64)
    ranks_sorted = np.empty(N, dtype=np.int64)
    i = 0

    while i < N:
        j = i + 1
        x = F_sorted[i, 0]
        y = F_sorted[i, 1]
        while j < N and F_sorted[j, 0] == x and F_sorted[j, 1] == y:
            j += 1

        bit_idx = y_codes[i] + 1
        best = 0
        query_idx = bit_idx
        while query_idx > 0:
            if bit[query_idx] > best:
                best = bit[query_idx]
            query_idx -= query_idx & -query_idx

        rank = best
        for pos in range(i, j):
            ranks_sorted[pos] = rank

        update_value = rank + 1
        update_idx = bit_idx
        while update_idx < bit.shape[0]:
            if update_value > bit[update_idx]:
                bit[update_idx] = update_value
            update_idx += update_idx & -update_idx

        i = j

    return ranks_sorted


def fast_non_dominated_sort_ranks_2d(F: np.ndarray) -> np.ndarray:
    N = F.shape[0]
    if N == 0:
        return np.empty(0, dtype=np.int64)

    order = np.lexsort((F[:, 1], F[:, 0]))
    F_sorted = F[order]
    _, y_codes = np.unique(F_sorted[:, 1], return_inverse=True)
    ranks_sorted = _fast_non_dominated_sort_ranks_2d_sorted(F_sorted, y_codes.astype(np.int64, copy=False))
    ranks = np.empty(N, dtype=np.int64)
    ranks[order] = ranks_sorted
    return ranks


@njit(cache=True)
def _fast_non_dominated_sort_ranks_general(F: np.ndarray) -> np.ndarray:
    N = F.shape[0]
    if N == 0:
        return np.empty(0, dtype=np.int64)

    M = F.shape[1]
    dominated_count = np.zeros(N, dtype=np.int64)
    out_degree = np.zeros(N, dtype=np.int64)

    for p in range(N - 1):
        for q in range(p + 1, N):
            p_dominates = True
            q_dominates = True
            p_strict = False
            q_strict = False

            for m in range(M):
                fp = F[p, m]
                fq = F[q, m]
                if fp < fq:
                    p_strict = True
                    q_dominates = False
                elif fp > fq:
                    q_strict = True
                    p_dominates = False
                if not p_dominates and not q_dominates:
                    break

            if p_dominates and p_strict:
                dominated_count[q] += 1
                out_degree[p] += 1
            elif q_dominates and q_strict:
                dominated_count[p] += 1
                out_degree[q] += 1

    offsets = np.empty(N + 1, dtype=np.int64)
    offsets[0] = 0
    for i in range(N):
        offsets[i + 1] = offsets[i] + out_degree[i]

    dominated = np.empty(offsets[N], dtype=np.int64)
    cursor = np.empty(N, dtype=np.int64)
    for i in range(N):
        cursor[i] = offsets[i]

    for p in range(N - 1):
        for q in range(p + 1, N):
            p_dominates = True
            q_dominates = True
            p_strict = False
            q_strict = False

            for m in range(M):
                fp = F[p, m]
                fq = F[q, m]
                if fp < fq:
                    p_strict = True
                    q_dominates = False
                elif fp > fq:
                    q_strict = True
                    p_dominates = False
                if not p_dominates and not q_dominates:
                    break

            if p_dominates and p_strict:
                dominated[cursor[p]] = q
                cursor[p] += 1
            elif q_dominates and q_strict:
                dominated[cursor[q]] = p
                cursor[q] += 1

    ranks = np.empty(N, dtype=np.int64)
    current = np.empty(N, dtype=np.int64)
    next_front = np.empty(N, dtype=np.int64)

    current_size = 0
    for i in range(N):
        if dominated_count[i] == 0:
            ranks[i] = 0
            current[current_size] = i
            current_size += 1

    level = 0
    while current_size > 0:
        next_size = 0
        for idx in range(current_size):
            p = current[idx]
            for edge_idx in range(offsets[p], offsets[p + 1]):
                q = dominated[edge_idx]
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    ranks[q] = level + 1
                    next_front[next_size] = q
                    next_size += 1

        for i in range(next_size):
            current[i] = next_front[i]
        current_size = next_size
        level += 1

    return ranks


def fast_non_dominated_sort_ranks(F: np.ndarray) -> np.ndarray:
    if F.shape[1] == 2:
        return fast_non_dominated_sort_ranks_2d(F)
    return _fast_non_dominated_sort_ranks_general(F)


@njit(cache=True)
def _stable_merge_sort_order(values: np.ndarray, order: np.ndarray, temp: np.ndarray, length: int) -> None:
    width = 1
    while width < length:
        left = 0
        while left < length:
            mid = left + width
            if mid > length:
                mid = length
            right = left + 2 * width
            if right > length:
                right = length

            i = left
            j = mid
            k = left

            while i < mid and j < right:
                left_idx = order[i]
                right_idx = order[j]
                left_val = values[left_idx]
                right_val = values[right_idx]

                if left_val <= right_val:
                    temp[k] = left_idx
                    i += 1
                else:
                    temp[k] = right_idx
                    j += 1
                k += 1

            while i < mid:
                temp[k] = order[i]
                i += 1
                k += 1

            while j < right:
                temp[k] = order[j]
                j += 1
                k += 1

            for pos in range(left, right):
                order[pos] = temp[pos]

            left += 2 * width

        width *= 2


@njit(cache=True)
def compute_crowding_numba(F: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    N = F.shape[0]
    crowding = np.zeros(N, dtype=np.float64)
    if N == 0:
        return crowding

    max_rank = int(ranks.max())
    counts = np.zeros(max_rank + 1, dtype=np.int64)
    for i in range(N):
        counts[ranks[i]] += 1

    offsets = np.empty(max_rank + 2, dtype=np.int64)
    offsets[0] = 0
    for rank in range(max_rank + 1):
        offsets[rank + 1] = offsets[rank] + counts[rank]

    front_indices = np.empty(N, dtype=np.int64)
    cursor = np.empty(max_rank + 1, dtype=np.int64)
    for rank in range(max_rank + 1):
        cursor[rank] = offsets[rank]

    for idx in range(N):
        rank = ranks[idx]
        front_indices[cursor[rank]] = idx
        cursor[rank] += 1

    max_front_size = int(counts.max())
    local_values = np.empty(max_front_size, dtype=np.float64)
    local_order = np.empty(max_front_size, dtype=np.int64)
    local_temp = np.empty(max_front_size, dtype=np.int64)
    local_dist = np.empty(max_front_size, dtype=np.float64)
    n_obj = F.shape[1]

    for rank in range(max_rank + 1):
        start = offsets[rank]
        end = offsets[rank + 1]
        front_size = end - start

        if front_size == 0:
            continue
        if front_size == 1:
            crowding[front_indices[start]] = np.inf
            continue

        for i in range(front_size):
            local_dist[i] = 0.0

        for obj in range(n_obj):
            for i in range(front_size):
                local_order[i] = i
                local_values[i] = F[front_indices[start + i], obj]

            _stable_merge_sort_order(local_values, local_order, local_temp, front_size)
            local_dist[local_order[0]] = np.inf
            local_dist[local_order[front_size - 1]] = np.inf

            span = local_values[local_order[front_size - 1]] - local_values[local_order[0]]
            if span <= 0.0:
                continue

            for pos in range(1, front_size - 1):
                local_dist[local_order[pos]] += (local_values[local_order[pos + 1]] - local_values[local_order[pos - 1]]) / span

        for i in range(front_size):
            crowding[front_indices[start + i]] = local_dist[i]

    return crowding


def fronts_from_ranks(ranks: np.ndarray) -> list[list[int]]:
    if ranks.size == 0:
        return []
    unique_ranks = np.unique(ranks)
    return [np.flatnonzero(ranks == rank).tolist() for rank in unique_ranks]


__all__ = [
    "compute_crowding_numba",
    "fast_non_dominated_sort_ranks",
    "fronts_from_ranks",
]
