# kernel/numba_backend.py
import numpy as np

try:
    from numba import njit
except ImportError as exc:  # pragma: no cover - informative error when numba missing
    raise ImportError("NumbaKernel requires the 'numba' package. Install it or switch to NumPyKernel.") from exc

from typing import Iterable

from .backend import KernelBackend
from .numpy_backend import NumPyKernel as _NumPyKernel


@njit(cache=True)
def _fast_non_dominated_sort_ranks(F: np.ndarray) -> np.ndarray:
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


class NumbaKernel(KernelBackend):
    """
    Alternative backend with critical kernels (ranking/survival) compiled with Numba.
    Stochastic operators (selection, crossover, mutation) reuse the NumPy implementations.
    """

    def __init__(self):
        self._numpy_ops = _NumPyKernel()

    def capabilities(self) -> Iterable[str]:
        return ("numba",)

    def quality_indicators(self) -> Iterable[str]:
        return ("hypervolume",)

    def nsga2_ranking(self, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ranks = _fast_non_dominated_sort_ranks(F)
        crowding = _compute_crowding_numba(F, ranks)
        return ranks.astype(np.int64), crowding

    def tournament_selection(
        self,
        ranks: np.ndarray,
        crowding: np.ndarray,
        pressure: int,
        rng: np.random.Generator,
        n_parents: int,
    ) -> np.ndarray:
        return self._numpy_ops.tournament_selection(ranks, crowding, pressure, rng, n_parents)

    def sbx_crossover(
        self,
        X_parents: np.ndarray,
        params: dict,
        rng: np.random.Generator,
        xl: float,
        xu: float,
    ) -> np.ndarray:
        return self._numpy_ops.sbx_crossover(X_parents, params, rng, xl, xu)

    def polynomial_mutation(
        self,
        X: np.ndarray,
        params: dict,
        rng: np.random.Generator,
        xl: float,
        xu: float,
    ) -> None:
        self._numpy_ops.polynomial_mutation(X, params, rng, xl, xu)

    def nsga2_survival(
        self,
        X: np.ndarray,
        F: np.ndarray,
        X_off: np.ndarray,
        F_off: np.ndarray,
        pop_size: int,
        return_indices: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_comb = np.vstack((X, X_off))
        F_comb = np.vstack((F, F_off))
        ranks = _fast_non_dominated_sort_ranks(F_comb)
        crowding = _compute_crowding_numba(F_comb, ranks)
        sel = _select_nsga2_indices(ranks, crowding, pop_size)
        if return_indices:
            return X_comb[sel], F_comb[sel], sel
        return X_comb[sel], F_comb[sel]

    def hypervolume(self, points: np.ndarray, reference_point: np.ndarray) -> float:
        from vamos.foundation.metrics.hypervolume import hypervolume as hv_fn

        return hv_fn(points, reference_point)
