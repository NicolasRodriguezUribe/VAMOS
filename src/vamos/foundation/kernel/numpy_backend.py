"""NumPy kernel backend.

Performance-sensitive: keep operations vectorized and avoid Python loops where possible.
Assumes F is float64 of shape (N, M), X is float64 of shape (N, n_var).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from vamos.operators.impl.real import SBXCrossover, PolynomialMutation

from .backend import KernelBackend


def _fast_non_dominated_sort(F: np.ndarray):
    """
    Classic O(N^2) fast non-dominated sort.
    Args:
        F: objective matrix (N, M), float64.
    Returns:
      - fronts: list of lists with indices per front (0, 1, ...)
      - rank: array with the front rank for each solution
    """
    N = F.shape[0]
    if N == 0:
        return [], np.empty(0, dtype=int)

    less_equal = F[:, None, :] <= F[None, :, :]
    strictly_less = F[:, None, :] < F[None, :, :]
    dom_matrix = np.logical_and(
        np.all(less_equal, axis=2),
        np.any(strictly_less, axis=2),
    )

    dominated_count = dom_matrix.sum(axis=0).astype(np.int64)
    rank = np.empty(N, dtype=int)
    fronts = []

    current = np.flatnonzero(dominated_count == 0)
    level = 0
    while current.size > 0:
        fronts.append(current.tolist())
        rank[current] = level
        dom_contrib = dom_matrix[current].sum(axis=0)
        dominated_count -= dom_contrib
        dominated_count[current] = -1
        dom_matrix[current] = False
        level += 1
        current = np.flatnonzero(dominated_count == 0)

    return fronts, rank


def _compute_crowding(F: np.ndarray, fronts):
    """
    Standard crowding-distance computation.
    crowding: array of length N.
    """
    N = F.shape[0]
    crowding = np.zeros(N)

    for front in fronts:
        if len(front) == 0:
            continue
        front_arr = np.asarray(front, dtype=int)
        if front_arr.size == 1:
            crowding[front_arr[0]] = np.inf
            continue

        fvals = F[front_arr]  # shape (k, n_obj)
        n_obj = fvals.shape[1]
        d = np.zeros(front_arr.size, dtype=float)

        for m in range(n_obj):
            order = np.argsort(fvals[:, m], kind="mergesort")
            sorted_vals = fvals[order, m]

            d[order[0]] = np.inf
            d[order[-1]] = np.inf

            span = sorted_vals[-1] - sorted_vals[0]
            if span <= 0.0:
                continue

            contrib = np.zeros_like(sorted_vals)
            contrib[1:-1] = (sorted_vals[2:] - sorted_vals[:-2]) / span
            d[order[1:-1]] += contrib[1:-1]

        crowding[front_arr] = d

    return crowding


def _select_nsga2(fronts, crowding: np.ndarray, pop_size: int) -> np.ndarray:
    """
    NSGA-II elitist selection based on fronts + crowding.
    """
    selected = []
    for front in fronts:
        if len(front) == 0:
            continue
        front_arr = np.asarray(front, dtype=int)
        if len(selected) + front_arr.size <= pop_size:
            selected.extend(front_arr.tolist())
        else:
            rem = pop_size - len(selected)
            order = np.argsort(crowding[front_arr])[::-1]
            selected.extend(front_arr[order[:rem]].tolist())
            break
    return np.array(selected, dtype=int)


class NumPyKernel(KernelBackend):
    """
    Backend with pure NumPy implementations of the NSGA-II kernels.
    """

    def __init__(self):
        self._row_index = np.empty(0, dtype=np.int64)
        self._score_buffer: np.ndarray | None = None

    def _ensure_row_index(self, length: int) -> np.ndarray:
        if self._row_index.shape[0] != length:
            self._row_index = np.arange(length, dtype=np.int64)
        return self._row_index

    def _ensure_score_buffer(self, shape: tuple[int, ...]) -> np.ndarray:
        if self._score_buffer is None or self._score_buffer.shape != shape:
            self._score_buffer = np.empty(shape, dtype=np.float64)
        return self._score_buffer

    @staticmethod
    def _normalize_bounds(
        xl: float | np.ndarray,
        xu: float | np.ndarray,
        n_var: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        lower = np.asarray(xl, dtype=float)
        upper = np.asarray(xu, dtype=float)
        if lower.ndim == 0 or (lower.ndim == 1 and lower.shape[0] == 1 and n_var > 1):
            lower = np.full(n_var, float(lower.reshape(-1)[0]))
        if upper.ndim == 0 or (upper.ndim == 1 and upper.shape[0] == 1 and n_var > 1):
            upper = np.full(n_var, float(upper.reshape(-1)[0]))
        return lower, upper

    def capabilities(self) -> Iterable[str]:
        return ("cpu",)

    def quality_indicators(self) -> Iterable[str]:
        return ("hypervolume",)

    def nsga2_ranking(self, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fronts, ranks = _fast_non_dominated_sort(F)
        crowding = _compute_crowding(F, fronts)
        return ranks, crowding

    def tournament_selection(
        self,
        ranks: np.ndarray,
        crowding: np.ndarray,
        pressure: int,
        rng: np.random.Generator,
        n_parents: int,
    ) -> np.ndarray:
        """
        Standard binary/m-ary tournament:
        smallest rank wins; break ties with higher crowding.
        """
        N = ranks.shape[0]
        if pressure <= 0:
            raise ValueError("pressure must be a positive integer")
        if n_parents <= 0 or N == 0:
            return np.empty(0, dtype=int)
        if pressure > N:
            raise ValueError("pressure cannot exceed population size for tournament selection without replacement")

        if pressure == 1:
            return rng.integers(0, N, size=n_parents, dtype=int)

        candidates = np.empty((n_parents, pressure), dtype=int)
        for i in range(n_parents):
            candidates[i] = rng.choice(N, size=pressure, replace=False)

        winners = np.empty(n_parents, dtype=int)
        for i in range(n_parents):
            row = candidates[i]
            row_ranks = ranks[row]
            min_rank = row_ranks.min()
            best = row[row_ranks == min_rank]
            if best.size == 1:
                winners[i] = int(best[0])
                continue
            best_crowd = np.nan_to_num(crowding[best], nan=-np.inf)
            max_crowd = best_crowd.max()
            tied = best[best_crowd == max_crowd]
            winners[i] = int(rng.choice(tied)) if tied.size > 1 else int(tied[0])
        return winners

    def sbx_crossover(
        self,
        X_parents: np.ndarray,
        params: dict,
        rng: np.random.Generator | None,
        xl: float,
        xu: float,
    ) -> np.ndarray:
        Np, D = X_parents.shape
        if Np == 0:
            return np.empty_like(X_parents)
        if rng is None:
            rng = np.random.default_rng()
        # Handle odd parent count by duplicating the last parent
        if Np % 2 != 0:
            X_parents = np.vstack([X_parents, X_parents[-1:]])
            Np += 1
        lower, upper = self._normalize_bounds(xl, xu, D)
        operator = SBXCrossover(
            prob_crossover=float(params.get("prob", 0.9)),
            eta=float(params.get("eta", 20.0)),
            lower=lower,
            upper=upper,
        )
        pairs = X_parents.reshape(Np // 2, 2, D)
        offspring = operator(pairs, rng)
        return offspring.reshape(Np, D)

    def polynomial_mutation(
        self,
        X: np.ndarray,
        params: dict,
        rng: np.random.Generator | None,
        xl: float,
        xu: float,
    ) -> None:
        if X.size == 0:
            return
        if rng is None:
            rng = np.random.default_rng()
        n_var = X.shape[1]
        lower, upper = self._normalize_bounds(xl, xu, n_var)
        operator = PolynomialMutation(
            prob_mutation=float(params.get("prob", 0.1)),
            eta=float(params.get("eta", 20.0)),
            lower=lower,
            upper=upper,
        )
        mutated = operator(X, rng)
        X[:] = mutated

    def nsga2_survival(
        self,
        X: np.ndarray,
        F: np.ndarray,
        X_off: np.ndarray,
        F_off: np.ndarray,
        pop_size: int,
        return_indices: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        NSGA-II elitism: merge parents + offspring, re-rank, and select.
        """
        X_comb = np.vstack([X, X_off])
        F_comb = np.vstack([F, F_off])
        fronts, _ = _fast_non_dominated_sort(F_comb)
        crowding = _compute_crowding(F_comb, fronts)
        sel = _select_nsga2(fronts, crowding, pop_size)
        if return_indices:
            return X_comb[sel], F_comb[sel], sel
        return X_comb[sel], F_comb[sel]

    def hypervolume(self, points: np.ndarray, reference_point: np.ndarray) -> float:
        from vamos.foundation.metrics.hypervolume import hypervolume as hv_fn

        return hv_fn(points, reference_point)
