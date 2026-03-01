# kernel/numba_backend.py
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, Literal, TypeVar, cast, overload

import numpy as np
from numba import njit as _numba_njit

from .backend import KernelBackend
from .numpy_backend import NumPyKernel as _NumPyKernel

_F = TypeVar("_F", bound=Callable[..., object])


def njit(*args: Any, **kwargs: Any) -> Callable[[_F], _F]:
    """Typed wrapper around numba.njit to keep mypy happy."""
    return cast(Callable[[_F], _F], _numba_njit(*args, **kwargs))


@njit(cache=True)
def _fast_non_dominated_sort_ranks_generic(F: np.ndarray) -> np.ndarray:
    N = F.shape[0]
    if N == 0:
        return np.empty(0, dtype=np.int64)

    M = F.shape[1]
    dom_matrix = np.zeros((N, N), dtype=np.bool_)
    dominated_count = np.zeros(N, dtype=np.int64)

    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            less_equal = True
            strictly_less = False
            for m in range(M):
                fp = F[p, m]
                fq = F[q, m]
                if fp > fq:
                    less_equal = False
                    break
                elif fp < fq:
                    strictly_less = True
            if less_equal and strictly_less:
                dom_matrix[p, q] = True
                dominated_count[q] += 1

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
            for q in range(N):
                if dom_matrix[p, q]:
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


@njit(cache=True)
def _sorted_indices_biobjective(F: np.ndarray) -> np.ndarray:
    N = F.shape[0]
    order = np.argsort(F[:, 0]).astype(np.int64)
    start = 0
    while start < N:
        end = start + 1
        f1 = F[order[start], 0]
        while end < N and F[order[end], 0] == f1:
            end += 1

        for i in range(start + 1, end):
            key = order[i]
            key_f2 = F[key, 1]
            j = i - 1
            while j >= start:
                prev = order[j]
                prev_f2 = F[prev, 1]
                if prev_f2 < key_f2 or (prev_f2 == key_f2 and prev <= key):
                    break
                order[j + 1] = prev
                j -= 1
            order[j + 1] = key
        start = end
    return order


@njit(cache=True)
def _fast_non_dominated_sort_ranks_biobjective(F: np.ndarray) -> np.ndarray:
    N = F.shape[0]
    ranks = np.empty(N, dtype=np.int64)
    if N == 0:
        return ranks

    order = _sorted_indices_biobjective(F)
    front_last_f1 = np.empty(N, dtype=np.float64)
    front_last_f2 = np.empty(N, dtype=np.float64)
    n_fronts = 0

    for pos in range(N):
        idx = order[pos]
        f1 = F[idx, 0]
        f2 = F[idx, 1]
        assigned = -1
        for front in range(n_fronts):
            last_f1 = front_last_f1[front]
            last_f2 = front_last_f2[front]
            if f2 < last_f2 or (f2 == last_f2 and f1 == last_f1):
                assigned = front
                break

        if assigned < 0:
            assigned = n_fronts
            n_fronts += 1

        ranks[idx] = assigned
        front_last_f1[assigned] = f1
        front_last_f2[assigned] = f2

    return ranks


@njit(cache=True)
def _fast_non_dominated_sort_ranks(F: np.ndarray) -> np.ndarray:
    if F.shape[1] == 2:
        return _fast_non_dominated_sort_ranks_biobjective(F)
    return _fast_non_dominated_sort_ranks_generic(F)


@njit(cache=True)
def _compute_crowding_numba(F: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    N = F.shape[0]
    crowding = np.zeros(N, dtype=np.float64)
    if N == 0:
        return crowding

    M = F.shape[1]
    max_rank = 0
    for i in range(N):
        if ranks[i] > max_rank:
            max_rank = ranks[i]

    front_idx = np.empty(N, dtype=np.int64)

    for r in range(max_rank + 1):
        size = 0
        for i in range(N):
            if ranks[i] == r:
                front_idx[size] = i
                size += 1

        if size == 0:
            continue
        if size == 1:
            crowding[front_idx[0]] = np.inf
            continue

        distances = np.zeros(size, dtype=np.float64)

        for m in range(M):
            values = np.empty(size, dtype=np.float64)
            for idx in range(size):
                values[idx] = F[front_idx[idx], m]

            order = np.argsort(values)
            distances[order[0]] = np.inf
            distances[order[-1]] = np.inf

            span = values[order[-1]] - values[order[0]]
            if span <= 0.0:
                continue

            for i in range(1, size - 1):
                distances[order[i]] += (values[order[i + 1]] - values[order[i - 1]]) / span

        for i in range(size):
            crowding[front_idx[i]] = distances[i]

    return crowding


@njit(cache=True)
def _select_nsga2_indices(ranks: np.ndarray, crowding: np.ndarray, pop_size: int) -> np.ndarray:
    N = ranks.shape[0]
    selected = np.empty(pop_size, dtype=np.int64)
    if pop_size == 0 or N == 0:
        return selected

    max_rank = 0
    for i in range(N):
        if ranks[i] > max_rank:
            max_rank = ranks[i]

    front_idx = np.empty(N, dtype=np.int64)
    taken = 0

    for r in range(max_rank + 1):
        size = 0
        for i in range(N):
            if ranks[i] == r:
                front_idx[size] = i
                size += 1

        if size == 0:
            continue

        if taken + size <= pop_size:
            for i in range(size):
                selected[taken + i] = front_idx[i]
            taken += size
            if taken == pop_size:
                break
        else:
            rem = pop_size - taken
            crowd_vals = np.empty(size, dtype=np.float64)
            for i in range(size):
                crowd_vals[i] = crowding[front_idx[i]]
            order = np.argsort(crowd_vals)
            for i in range(rem):
                selected[taken + i] = front_idx[order[size - 1 - i]]
            taken += rem
            break

    return selected


@njit(cache=True)
def _crowding_tournament_value(value: float) -> float:
    if np.isnan(value):
        return -np.inf
    return value


@njit(cache=True)
def _tournament_winners_from_candidates(
    ranks: np.ndarray,
    crowding: np.ndarray,
    candidates: np.ndarray,
    tie_breaks: np.ndarray,
) -> np.ndarray:
    n_parents = candidates.shape[0]
    pressure = candidates.shape[1]
    winners = np.empty(n_parents, dtype=np.int64)
    for i in range(n_parents):
        best = candidates[i, 0]
        best_rank = ranks[best]
        best_crowd = _crowding_tournament_value(crowding[best])
        n_ties = 1
        for j in range(1, pressure):
            cand = candidates[i, j]
            cand_rank = ranks[cand]
            cand_crowd = _crowding_tournament_value(crowding[cand])
            if cand_rank < best_rank or (cand_rank == best_rank and cand_crowd > best_crowd):
                best = cand
                best_rank = cand_rank
                best_crowd = cand_crowd
                n_ties = 1
            elif cand_rank == best_rank and cand_crowd == best_crowd:
                n_ties += 1
                if tie_breaks[i, j] < (1.0 / n_ties):
                    best = cand
        winners[i] = best
    return winners


class NumbaKernel(KernelBackend):
    """
    Alternative backend with critical kernels (ranking/survival) compiled with Numba.
    Stochastic operators (selection, crossover, mutation) reuse the NumPy implementations.
    """

    name = "numba"

    def __init__(self) -> None:
        self._numpy_ops = _NumPyKernel()
        self._x_comb_buffer: np.ndarray | None = None
        self._f_comb_buffer: np.ndarray | None = None

    def _combine_rows(self, left: np.ndarray, right: np.ndarray, *, attr_name: str) -> np.ndarray:
        dtype = np.result_type(left.dtype, right.dtype)
        shape = (left.shape[0] + right.shape[0], left.shape[1])
        buffer = cast(np.ndarray | None, getattr(self, attr_name))
        if buffer is None or buffer.shape != shape or buffer.dtype != np.dtype(dtype):
            buffer = np.empty(shape, dtype=dtype)
            setattr(self, attr_name, buffer)
        buffer[: left.shape[0]] = left
        buffer[left.shape[0] :] = right
        return buffer

    @staticmethod
    def _sample_tournament_candidates(
        population_size: int,
        pressure: int,
        n_parents: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if pressure == 2:
            candidates = np.empty((n_parents, 2), dtype=np.int64)
            candidates[:, 0] = rng.integers(0, population_size, size=n_parents, dtype=np.int64)
            candidates[:, 1] = rng.integers(0, population_size, size=n_parents, dtype=np.int64)
            duplicates = candidates[:, 0] == candidates[:, 1]
            while np.any(duplicates):
                candidates[duplicates, 1] = rng.integers(0, population_size, size=int(np.sum(duplicates)), dtype=np.int64)
                duplicates = candidates[:, 0] == candidates[:, 1]
            return candidates

        candidates = np.empty((n_parents, pressure), dtype=np.int64)
        for i in range(n_parents):
            candidates[i] = rng.choice(population_size, size=pressure, replace=False)
        return candidates

    def capabilities(self) -> Iterable[str]:
        return ("numba",)

    def quality_indicators(self) -> Iterable[str]:
        return ("hypervolume",)

    def nsga2_ranking(self, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ranks = _fast_non_dominated_sort_ranks(F)
        crowding = _compute_crowding_numba(F, ranks)
        return ranks, crowding

    def tournament_selection(
        self,
        ranks: np.ndarray,
        crowding: np.ndarray,
        pressure: int,
        rng: np.random.Generator,
        n_parents: int,
    ) -> np.ndarray:
        population_size = ranks.shape[0]
        if pressure <= 0:
            raise ValueError("pressure must be a positive integer")
        if n_parents <= 0 or population_size == 0:
            return np.empty(0, dtype=np.int64)
        if pressure > population_size:
            raise ValueError("pressure cannot exceed population size for tournament selection without replacement")
        if pressure == 1:
            return rng.integers(0, population_size, size=n_parents, dtype=np.int64)

        candidates = self._sample_tournament_candidates(population_size, pressure, n_parents, rng)
        tie_breaks = rng.random(candidates.shape)
        return _tournament_winners_from_candidates(ranks, crowding, candidates, tie_breaks)

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
        if rng is None:
            rng = np.random.default_rng()
        self._numpy_ops.polynomial_mutation(X, params, rng, xl, xu)

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
        X_comb = self._combine_rows(X, X_off, attr_name="_x_comb_buffer")
        F_comb = self._combine_rows(F, F_off, attr_name="_f_comb_buffer")
        ranks = _fast_non_dominated_sort_ranks(F_comb)
        crowding = _compute_crowding_numba(F_comb, ranks)
        sel = _select_nsga2_indices(ranks, crowding, pop_size)
        if return_indices:
            return X_comb[sel], F_comb[sel], sel
        return X_comb[sel], F_comb[sel]

    def hypervolume(self, points: np.ndarray, reference_point: np.ndarray) -> float:
        from vamos.foundation.quality_indicators.hypervolume import hypervolume as hv_fn

        return hv_fn(points, reference_point)
