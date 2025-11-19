from __future__ import annotations

from typing import Iterable

import numpy as np

from vamos.operators.real import SBXCrossover, PolynomialMutation

from .backend import KernelBackend

def _fast_non_dominated_sort(F: np.ndarray):
    """
    Classic O(N^2) fast non-dominated sort.
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

        candidates = rng.integers(0, N, size=(n_parents, pressure))
        candidate_ranks = ranks[candidates]
        candidate_crowding = crowding[candidates]

        finite_crowding = candidate_crowding[np.isfinite(candidate_crowding)]
        crowd_span = float(finite_crowding.max() - finite_crowding.min()) if finite_crowding.size else 0.0
        rank_scale = crowd_span + 1.0
        if not np.isfinite(rank_scale):
            rank_scale = 1.0

        scores = self._ensure_score_buffer(candidate_ranks.shape)
        np.multiply(candidate_ranks, rank_scale, out=scores, casting="unsafe")
        scores -= candidate_crowding

        winner_cols = np.argmin(scores, axis=1)
        row_idx = self._ensure_row_index(n_parents)
        return candidates[row_idx, winner_cols]

    def sbx_crossover(
        self,
        X_parents: np.ndarray,
        params: dict,
        rng: np.random.Generator,
        xl: float,
        xu: float,
    ) -> np.ndarray:
        Np, D = X_parents.shape
        if Np == 0:
            return np.empty_like(X_parents)
        if Np % 2 != 0:
            raise ValueError("SBX crossover expects an even number of parents.")
        operator = SBXCrossover(
            prob_crossover=float(params.get("prob", 0.9)),
            eta=float(params.get("eta", 20.0)),
            lower=xl,
            upper=xu,
        )
        pairs = X_parents.reshape(Np // 2, 2, D)
        offspring = operator(pairs, rng)
        return offspring.reshape(Np, D)

    def polynomial_mutation(
        self,
        X: np.ndarray,
        params: dict,
        rng: np.random.Generator,
        xl: float,
        xu: float,
    ) -> None:
        if X.size == 0:
            return
        operator = PolynomialMutation(
            prob_mutation=float(params.get("prob", 0.1)),
            eta=float(params.get("eta", 20.0)),
            lower=xl,
            upper=xu,
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
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        NSGA-II elitism: merge parents + offspring, re-rank, and select.
        """
        X_comb = np.vstack([X, X_off])
        F_comb = np.vstack([F, F_off])
        fronts, _ = _fast_non_dominated_sort(F_comb)
        crowding = _compute_crowding(F_comb, fronts)
        sel = _select_nsga2(fronts, crowding, pop_size)
        return X_comb[sel], F_comb[sel]

    def hypervolume(self, points: np.ndarray, reference_point: np.ndarray) -> float:
        from vamos.algorithm.hypervolume import hypervolume as hv_fn

        return hv_fn(points, reference_point)
