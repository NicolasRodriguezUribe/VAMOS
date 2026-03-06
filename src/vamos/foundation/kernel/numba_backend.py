# kernel/numba_backend.py
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, Literal, TypeVar, cast, overload

import numpy as np
from numba import njit as _numba_njit

from .backend import KernelBackend
from .numba_ops import _polynomial_mutation_numba_impl
from .numpy_backend import NumPyKernel as _NumPyKernel
from .numpy_backend import _as_float, _select_nsga2

_F = TypeVar("_F", bound=Callable[..., object])


def njit(*args: Any, **kwargs: Any) -> Callable[[_F], _F]:
    """Typed wrapper around numba.njit to keep mypy happy."""
    return cast(Callable[[_F], _F], _numba_njit(*args, **kwargs))


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


def _fast_non_dominated_sort_ranks_2d(F: np.ndarray) -> np.ndarray:
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


def _fast_non_dominated_sort_ranks(F: np.ndarray) -> np.ndarray:
    if F.shape[1] == 2:
        return _fast_non_dominated_sort_ranks_2d(F)
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

                # Preserve NumPy mergesort semantics by taking the left element on ties.
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
def _compute_crowding_numba(F: np.ndarray, ranks: np.ndarray) -> np.ndarray:
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

    # Group each front in ascending solution-index order to match the NumPy path.
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


@njit(cache=True)
def _binary_tournament_winners_numba(
    ranks: np.ndarray,
    crowding: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    tie_break: np.ndarray,
) -> np.ndarray:
    n_parents = first.shape[0]
    winners = np.empty(n_parents, dtype=np.int64)

    for i in range(n_parents):
        left = first[i]
        right = second[i]
        left_rank = ranks[left]
        right_rank = ranks[right]

        if left_rank < right_rank:
            winners[i] = left
            continue
        if right_rank < left_rank:
            winners[i] = right
            continue

        left_crowding = crowding[left]
        right_crowding = crowding[right]

        if np.isnan(left_crowding):
            left_crowding = -np.inf
        if np.isnan(right_crowding):
            right_crowding = -np.inf

        if left_crowding > right_crowding:
            winners[i] = left
            continue
        if right_crowding > left_crowding:
            winners[i] = right
            continue

        winners[i] = left if tie_break[i] == 0 else right

    return winners


def _fronts_from_ranks(ranks: np.ndarray) -> list[list[int]]:
    if ranks.size == 0:
        return []
    unique_ranks = np.unique(ranks)
    return [np.flatnonzero(ranks == rank).tolist() for rank in unique_ranks]


class NumbaKernel(KernelBackend):
    """
    Alternative backend with critical kernels (ranking/survival) compiled with Numba.
    Binary tournament selection and mutation use Numba-backed paths.
    Crossover still reuses the NumPy implementation.
    Mutation uses the Numba kernel, but randomness still comes from the caller RNG.
    """

    name = "numba"

    def __init__(self) -> None:
        self._numpy_ops = _NumPyKernel()
        self._X_buffer: np.ndarray | None = None
        self._F_buffer: np.ndarray | None = None
        self._X_output: np.ndarray | None = None
        self._F_output: np.ndarray | None = None
        self._mutation_mask: np.ndarray | None = None
        self._mutation_delta: np.ndarray | None = None

    def _ensure_buffers(self, total: int, n_var: int, n_obj: int, dtype: Any) -> None:
        dtype = np.dtype(dtype)
        if self._X_buffer is None or self._X_buffer.shape[0] < total or self._X_buffer.shape[1] != n_var or self._X_buffer.dtype != dtype:
            self._X_buffer = np.empty((total, n_var), dtype=dtype, order="C")
        if self._F_buffer is None or self._F_buffer.shape[0] < total or self._F_buffer.shape[1] != n_obj:
            self._F_buffer = np.empty((total, n_obj), dtype=np.float64, order="C")

    def _combine(
        self,
        X: np.ndarray,
        F: np.ndarray,
        X_off: np.ndarray,
        F_off: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        total = X.shape[0] + X_off.shape[0]
        self._ensure_buffers(total, X.shape[1], F.shape[1], X.dtype)
        assert self._X_buffer is not None
        assert self._F_buffer is not None
        X_view = self._X_buffer[:total]
        F_view = self._F_buffer[:total]
        split = X.shape[0]
        np.copyto(X_view[:split], X)
        np.copyto(F_view[:split], F, casting="unsafe")
        np.copyto(X_view[split:], X_off)
        np.copyto(F_view[split:], F_off, casting="unsafe")
        return X_view, F_view

    def _ensure_output_buffers(self, size: int, n_var: int, n_obj: int, dtype: Any) -> None:
        dtype = np.dtype(dtype)
        if self._X_output is None or self._X_output.shape[0] < size or self._X_output.shape[1] != n_var or self._X_output.dtype != dtype:
            self._X_output = np.empty((size, n_var), dtype=dtype, order="C")
        if self._F_output is None or self._F_output.shape[0] < size or self._F_output.shape[1] != n_obj:
            self._F_output = np.empty((size, n_obj), dtype=np.float64, order="C")

    def _ensure_mutation_buffers(self, n_ind: int, n_var: int) -> None:
        if self._mutation_mask is None or self._mutation_mask.shape[0] < n_ind or self._mutation_mask.shape[1] != n_var:
            self._mutation_mask = np.empty((n_ind, n_var), dtype=np.float64, order="C")
        if self._mutation_delta is None or self._mutation_delta.shape[0] < n_ind or self._mutation_delta.shape[1] != n_var:
            self._mutation_delta = np.empty((n_ind, n_var), dtype=np.float64, order="C")

    def _rank_and_crowding(self, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        F_arr = np.asarray(F, dtype=np.float64, order="C")
        if not np.isfinite(F_arr).all():
            # Preserve NumPy semantics on invalid fronts instead of diverging in JIT code.
            ranks, crowding = self._numpy_ops.nsga2_ranking(F_arr)
            return ranks.astype(np.int64, copy=False), crowding
        ranks = _fast_non_dominated_sort_ranks(F_arr)
        crowding = _compute_crowding_numba(F_arr, ranks)
        return ranks.astype(np.int64, copy=False), crowding

    def capabilities(self) -> Iterable[str]:
        return ("numba",)

    def quality_indicators(self) -> Iterable[str]:
        return ("hypervolume",)

    def nsga2_ranking(self, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._rank_and_crowding(F)

    def tournament_selection(
        self,
        ranks: np.ndarray,
        crowding: np.ndarray,
        pressure: int,
        rng: np.random.Generator,
        n_parents: int,
    ) -> np.ndarray:
        n_candidates = ranks.shape[0]
        if pressure <= 0:
            raise ValueError("pressure must be a positive integer")
        if n_parents <= 0 or n_candidates == 0:
            return np.empty(0, dtype=int)
        if pressure > n_candidates:
            raise ValueError("pressure cannot exceed population size for tournament selection without replacement")

        if pressure == 1:
            return rng.integers(0, n_candidates, size=n_parents, dtype=int)

        if pressure == 2:
            ranks_arr = np.asarray(ranks, dtype=np.int64)
            crowding_arr = np.asarray(crowding, dtype=np.float64)
            first = rng.integers(0, n_candidates, size=n_parents, dtype=np.int64)
            second = rng.integers(0, n_candidates - 1, size=n_parents, dtype=np.int64)
            second += second >= first
            tie_break = rng.integers(0, 2, size=n_parents, dtype=np.int64)
            return _binary_tournament_winners_numba(ranks_arr, crowding_arr, first, second, tie_break)

        return self._numpy_ops.tournament_selection(ranks, crowding, pressure, rng, n_parents)

    def sbx_crossover(
        self,
        X_parents: np.ndarray,
        params: Mapping[str, object],
        rng: np.random.Generator | None,
        xl: float,
        xu: float,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        return self._numpy_ops.sbx_crossover(X_parents, params, rng, xl, xu)

    def polynomial_mutation(
        self,
        X: np.ndarray,
        params: Mapping[str, object],
        rng: np.random.Generator | None,
        xl: float,
        xu: float,
    ) -> None:
        if X.size == 0:
            return
        if rng is None:
            rng = np.random.default_rng()
        n_ind, n_var = X.shape
        lower, upper = self._numpy_ops._normalize_bounds(xl, xu, n_var)
        self._ensure_mutation_buffers(n_ind, n_var)
        assert self._mutation_mask is not None
        assert self._mutation_delta is not None
        rnd_mask = self._mutation_mask[:n_ind, :n_var]
        rnd_delta = self._mutation_delta[:n_ind, :n_var]
        rng.random(out=rnd_mask)
        rng.random(out=rnd_delta)
        _polynomial_mutation_numba_impl(
            X,
            _as_float(params.get("prob"), 0.1),
            _as_float(params.get("eta"), 20.0),
            lower,
            upper,
            rnd_mask,
            rnd_delta,
        )

    @overload
    def nsga2_survival(
        self,
        X: np.ndarray,
        F: np.ndarray,
        X_off: np.ndarray,
        F_off: np.ndarray,
        pop_size: int,
        return_indices: Literal[False] = False,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    @overload
    def nsga2_survival(
        self,
        X: np.ndarray,
        F: np.ndarray,
        X_off: np.ndarray,
        F_off: np.ndarray,
        pop_size: int,
        return_indices: Literal[True],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    def nsga2_survival(
        self,
        X: np.ndarray,
        F: np.ndarray,
        X_off: np.ndarray,
        F_off: np.ndarray,
        pop_size: int,
        return_indices: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_comb, F_comb = self._combine(X, F, X_off, F_off)
        if not np.isfinite(F_comb).all():
            return self._numpy_ops.nsga2_survival(X, F, X_off, F_off, pop_size, return_indices=return_indices)
        ranks, crowding = self._rank_and_crowding(F_comb)
        sel = _select_nsga2(_fronts_from_ranks(ranks), crowding, pop_size)
        self._ensure_output_buffers(pop_size, X_comb.shape[1], F_comb.shape[1], X_comb.dtype)
        assert self._X_output is not None
        assert self._F_output is not None
        np.copyto(self._X_output[:pop_size], X_comb[sel])
        np.copyto(self._F_output[:pop_size], F_comb[sel])
        if return_indices:
            return self._X_output[:pop_size], self._F_output[:pop_size], sel
        return self._X_output[:pop_size], self._F_output[:pop_size]

    def hypervolume(self, points: np.ndarray, reference_point: np.ndarray) -> float:
        from vamos.foundation.quality_indicators.hypervolume import hypervolume as hv_fn

        return hv_fn(points, reference_point)
