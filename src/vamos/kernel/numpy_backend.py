from __future__ import annotations

from typing import Iterable

import numpy as np

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

        best_rank = candidate_ranks.min(axis=1, keepdims=True)
        best_mask = candidate_ranks == best_rank

        candidate_crowding = crowding[candidates]
        tie_break = np.where(best_mask, candidate_crowding, -np.inf)

        winner_cols = np.argmax(tie_break, axis=1)
        row_idx = np.arange(n_parents)
        return candidates[row_idx, winner_cols]

    def sbx_crossover(
        self,
        X_parents: np.ndarray,
        params: dict,
        rng: np.random.Generator,
        xl: float,
        xu: float,
    ) -> np.ndarray:
        eta = float(params.get("eta", 20.0))
        prob = float(params.get("prob", 0.9))
        Np, D = X_parents.shape
        if Np == 0:
            return np.empty_like(X_parents)
        assert Np % 2 == 0, "SBX expects an even number of parents."
        n_pairs = Np // 2
        parents = X_parents.reshape(n_pairs, 2, D)
        offspring = parents.copy()
        eps = 1.0e-14
        pair_mask = rng.random(n_pairs) <= prob
        if not np.any(pair_mask):
            return offspring.reshape(Np, D)
        var_mask = rng.random((n_pairs, D)) <= 0.5
        diff_mask = np.abs(parents[:, 0, :] - parents[:, 1, :]) > eps
        active_mask = pair_mask[:, None] & var_mask & diff_mask
        idx_row, idx_col = np.nonzero(active_mask)
        if idx_row.size == 0:
            return offspring.reshape(Np, D)
        x1 = parents[idx_row, 0, idx_col]
        x2 = parents[idx_row, 1, idx_col]
        y1 = np.minimum(x1, x2)
        y2 = np.maximum(x1, x2)
        delta = y2 - y1
        xl_arr = np.full(idx_row.shape, xl, dtype=float)
        xu_arr = np.full(idx_row.shape, xu, dtype=float)
        rand = rng.random(idx_row.shape)
        inv = 1.0 / (eta + 1.0)

        beta = 1.0 + (2.0 * (y1 - xl_arr) / delta)
        beta = np.maximum(beta, eps)
        alpha = 2.0 - np.power(beta, -(eta + 1.0))
        term = rand <= (1.0 / alpha)
        betaq = np.empty_like(y1)
        betaq[term] = np.power(rand[term] * alpha[term], inv)
        betaq[~term] = np.power(1.0 / (2.0 - rand[~term] * alpha[~term]), inv)
        c1 = 0.5 * ((y1 + y2) - betaq * delta)

        beta = 1.0 + (2.0 * (xu_arr - y2) / delta)
        beta = np.maximum(beta, eps)
        alpha = 2.0 - np.power(beta, -(eta + 1.0))
        term = rand <= (1.0 / alpha)
        betaq[term] = np.power(rand[term] * alpha[term], inv)
        betaq[~term] = np.power(1.0 / (2.0 - rand[~term] * alpha[~term]), inv)
        c2 = 0.5 * ((y1 + y2) + betaq * delta)

        c1 = np.clip(c1, xl, xu)
        c2 = np.clip(c2, xl, xu)

        swap_mask = rng.random(idx_row.shape) <= 0.5
        first = np.where(swap_mask, c2, c1)
        second = np.where(swap_mask, c1, c2)
        offspring[idx_row, 0, idx_col] = first
        offspring[idx_row, 1, idx_col] = second
        return offspring.reshape(Np, D)

    def polynomial_mutation(
        self,
        X: np.ndarray,
        params: dict,
        rng: np.random.Generator,
        xl: float,
        xu: float,
    ) -> None:
        eta = float(params.get("eta", 20.0))
        p_mut = float(params.get("prob", 0.1))
        N, D = X.shape
        if p_mut <= 0.0 or N == 0:
            return
        xl_arr = np.asarray(xl, dtype=float)
        xu_arr = np.asarray(xu, dtype=float)
        if xl_arr.ndim == 0:
            xl_arr = np.full(D, xl_arr)
        if xu_arr.ndim == 0:
            xu_arr = np.full(D, xu_arr)
        mask = rng.random((N, D)) <= p_mut
        if not np.any(mask):
            return
        rows, cols = np.nonzero(mask)
        yl = xl_arr[cols]
        yu = xu_arr[cols]
        span = yu - yl
        y = X[rows, cols]
        delta1 = (y - yl) / span
        delta2 = (yu - y) / span
        rnd = rng.random(y.shape)
        mut_pow = 1.0 / (eta + 1.0)
        deltaq = np.empty_like(y)
        mask_lower = rnd <= 0.5
        if np.any(mask_lower):
            xy = 1.0 - delta1[mask_lower]
            val = 2.0 * rnd[mask_lower] + (1.0 - 2.0 * rnd[mask_lower]) * (xy ** (eta + 1.0))
            deltaq[mask_lower] = val ** mut_pow - 1.0
        mask_upper = ~mask_lower
        if np.any(mask_upper):
            xy = 1.0 - delta2[mask_upper]
            val = 2.0 * (1.0 - rnd[mask_upper]) + 2.0 * (rnd[mask_upper] - 0.5) * (xy ** (eta + 1.0))
            deltaq[mask_upper] = 1.0 - val ** mut_pow
        y += deltaq * span
        y = np.clip(y, yl, yu)
        X[rows, cols] = y

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
