# kernel/numpy_backend.py
import numpy as np

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


class NumPyKernel:
    """
    Backend with pure NumPy implementations of the NSGA-II kernels.
    """

    # ---------- Ranking / crowding ----------

    @staticmethod
    def nsga2_ranking(F: np.ndarray):
        fronts, ranks = _fast_non_dominated_sort(F)
        crowding = _compute_crowding(F, fronts)
        return ranks, crowding

    # ---------- Tournament selection ----------

    @staticmethod
    def tournament_selection(
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

    # ---------- SBX (Simulated Binary Crossover) ----------

    @staticmethod
    def sbx_crossover(
        X_parents: np.ndarray,
        params: dict,
        rng: np.random.Generator,
        xl: float,
        xu: float,
    ) -> np.ndarray:
        """
        X_parents: shape (2 * n_pairs, D)
        Returns offspring with the same shape.
        """
        eta = params["eta"]
        prob = params["prob"]

        Np, D = X_parents.shape
        assert Np % 2 == 0
        n_pairs = Np // 2

        if n_pairs == 0:
            return np.empty_like(X_parents)

        parents = X_parents.reshape(n_pairs, 2, D)
        offspring = parents.copy()

        if prob > 0.0:
            cross_mask = rng.random(n_pairs) < prob
            idx = np.flatnonzero(cross_mask)
            if idx.size > 0:
                u = rng.random((idx.size, D))
                beta = np.empty_like(u)
                lower = u <= 0.5
                beta[lower] = (2.0 * u[lower]) ** (1.0 / (eta + 1.0))
                beta[~lower] = (2.0 * (1.0 - u[~lower])) ** (-1.0 / (eta + 1.0))

                p1 = parents[idx, 0]
                p2 = parents[idx, 1]
                child1 = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
                child2 = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)

                offspring[idx, 0] = child1
                offspring[idx, 1] = child2

        np.clip(offspring, xl, xu, out=offspring)
        return offspring.reshape(Np, D)

    # ---------- Polynomial mutation ----------

    @staticmethod
    def polynomial_mutation(
        X: np.ndarray,
        params: dict,
        rng: np.random.Generator,
        xl: float,
        xu: float,
    ) -> None:
        """
        Standard polynomial mutation, in-place.
        Uses the same xl/xu bounds for every variable.
        """
        eta = params["eta"]
        p_mut = params["prob"]
        N, D = X.shape

        if p_mut <= 0.0:
            return

        mutation_mask = rng.random((N, D)) < p_mut
        if not np.any(mutation_mask):
            return

        y = X[mutation_mask]
        yl = xl
        yu = xu
        if yl == yu:
            return
        delta = yu - yl

        delta1 = (y - yl) / delta
        delta2 = (yu - y) / delta
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

        y = y + deltaq * delta
        np.clip(y, yl, yu, out=y)
        X[mutation_mask] = y

    # ---------- NSGA-II survival ----------

    @staticmethod
    def nsga2_survival(
        X: np.ndarray,
        F: np.ndarray,
        X_off: np.ndarray,
        F_off: np.ndarray,
        pop_size: int,
    ):
        """
        NSGA-II elitism: merge parents + offspring, re-rank, and select.
        """
        X_comb = np.vstack([X, X_off])
        F_comb = np.vstack([F, F_off])
        fronts, _ = _fast_non_dominated_sort(F_comb)
        crowding = _compute_crowding(F_comb, fronts)
        sel = _select_nsga2(fronts, crowding, pop_size)
        return X_comb[sel], F_comb[sel]
