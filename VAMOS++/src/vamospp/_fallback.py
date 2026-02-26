from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, overload

import numpy as np


def _require_float64_c(name: str, value: np.ndarray, ndim: int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype != np.float64:
        raise TypeError(f"{name} must have dtype=float64.")
    if arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D.")
    if not arr.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous.")
    return arr


def _require_int64_c(name: str, value: np.ndarray, ndim: int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype != np.int64:
        raise TypeError(f"{name} must have dtype=int64.")
    if arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D.")
    if not arr.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous.")
    return arr


def _as_uint64_seed(seed: int | np.integer[int]) -> int:
    return int(np.uint64(seed))


def _normalize_bounds(xl: np.ndarray, xu: np.ndarray, n_var: int) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(xl, dtype=np.float64)
    upper = np.asarray(xu, dtype=np.float64)
    if lower.ndim == 0 or (lower.ndim == 1 and lower.shape[0] == 1 and n_var > 1):
        lower = np.full(n_var, float(lower.reshape(-1)[0]), dtype=np.float64)
    if upper.ndim == 0 or (upper.ndim == 1 and upper.shape[0] == 1 and n_var > 1):
        upper = np.full(n_var, float(upper.reshape(-1)[0]), dtype=np.float64)
    if lower.shape != (n_var,) or upper.shape != (n_var,):
        raise ValueError("Bounds must be scalar-broadcastable or shape (n_var,).")
    return lower, upper


def fast_non_dominated_sort(F: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
    F = _require_float64_c("F", np.asarray(F), ndim=2)
    n = F.shape[0]
    if n == 0:
        return [], np.empty(0, dtype=np.int64)

    less_equal = F[:, None, :] <= F[None, :, :]
    strictly_less = F[:, None, :] < F[None, :, :]
    dom_matrix = np.logical_and(np.all(less_equal, axis=2), np.any(strictly_less, axis=2))

    dominated_count = dom_matrix.sum(axis=0).astype(np.int64)
    rank = np.empty(n, dtype=np.int64)
    fronts: list[list[int]] = []

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


def crowding_distance(F: np.ndarray, fronts: list[list[int]] | None = None) -> np.ndarray:
    F = _require_float64_c("F", np.asarray(F), ndim=2)
    if fronts is None:
        fronts, _ = fast_non_dominated_sort(F)

    n = F.shape[0]
    crowding = np.zeros(n, dtype=np.float64)
    for front in fronts:
        if not front:
            continue
        idx = np.asarray(front, dtype=np.int64)
        if idx.size == 1:
            crowding[idx[0]] = np.inf
            continue
        vals = F[idx]
        n_obj = vals.shape[1]
        d = np.zeros(idx.size, dtype=np.float64)

        for m in range(n_obj):
            order = np.argsort(vals[:, m], kind="mergesort")
            sorted_vals = vals[order, m]

            d[order[0]] = np.inf
            d[order[-1]] = np.inf
            span = sorted_vals[-1] - sorted_vals[0]
            if span <= 0.0:
                continue
            contrib = np.zeros_like(sorted_vals)
            contrib[1:-1] = (sorted_vals[2:] - sorted_vals[:-2]) / span
            d[order[1:-1]] += contrib[1:-1]

        crowding[idx] = d
    return crowding


def nsga2_ranking(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fronts, ranks = fast_non_dominated_sort(F)
    crowd = crowding_distance(F, fronts)
    return ranks, crowd


def _tournament_selection_rng(
    ranks: np.ndarray,
    crowding: np.ndarray,
    pressure: int,
    rng: np.random.Generator,
    n_parents: int,
) -> np.ndarray:
    n = int(ranks.shape[0])
    if pressure <= 0:
        raise ValueError("pressure must be positive.")
    if n_parents <= 0 or n == 0:
        return np.empty(0, dtype=np.int64)
    if pressure > n:
        raise ValueError("pressure cannot exceed population size.")
    if pressure == 1:
        return rng.integers(0, n, size=n_parents, dtype=np.int64)

    candidates = np.empty((n_parents, pressure), dtype=np.int64)
    for i in range(n_parents):
        candidates[i] = rng.choice(n, size=pressure, replace=False)

    winners = np.empty(n_parents, dtype=np.int64)
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


def tournament_selection(
    ranks: np.ndarray,
    crowding: np.ndarray,
    pressure: int,
    seed: int,
    n_parents: int,
) -> np.ndarray:
    ranks_arr = _require_int64_c("ranks", np.asarray(ranks), ndim=1)
    crowd_arr = _require_float64_c("crowding", np.asarray(crowding), ndim=1)
    if ranks_arr.shape != crowd_arr.shape:
        raise ValueError("ranks and crowding must have the same shape.")
    rng = np.random.default_rng(_as_uint64_seed(seed))
    return _tournament_selection_rng(ranks_arr, crowd_arr, int(pressure), rng, int(n_parents))


def _select_nsga2(fronts: list[list[int]], crowding: np.ndarray, pop_size: int) -> np.ndarray:
    selected: list[int] = []
    for front in fronts:
        if not front:
            continue
        front_arr = np.asarray(front, dtype=np.int64)
        if len(selected) + front_arr.size <= pop_size:
            selected.extend(front_arr.tolist())
        else:
            rem = pop_size - len(selected)
            order = np.argsort(crowding[front_arr])[::-1]
            selected.extend(front_arr[order[:rem]].tolist())
            break
    return np.asarray(selected, dtype=np.int64)


@overload
def nsga2_survival(
    X: np.ndarray,
    F: np.ndarray,
    X_off: np.ndarray,
    F_off: np.ndarray,
    pop_size: int,
    return_indices: Literal[False] = False,
) -> tuple[np.ndarray, np.ndarray]: ...


@overload
def nsga2_survival(
    X: np.ndarray,
    F: np.ndarray,
    X_off: np.ndarray,
    F_off: np.ndarray,
    pop_size: int,
    return_indices: Literal[True] = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


def nsga2_survival(
    X: np.ndarray,
    F: np.ndarray,
    X_off: np.ndarray,
    F_off: np.ndarray,
    pop_size: int,
    return_indices: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = _require_float64_c("X", np.asarray(X), ndim=2)
    F = _require_float64_c("F", np.asarray(F), ndim=2)
    X_off = _require_float64_c("X_off", np.asarray(X_off), ndim=2)
    F_off = _require_float64_c("F_off", np.asarray(F_off), ndim=2)

    X_comb = np.vstack([X, X_off])
    F_comb = np.vstack([F, F_off])
    fronts, _ = fast_non_dominated_sort(F_comb)
    crowd = crowding_distance(F_comb, fronts)
    sel = _select_nsga2(fronts, crowd, int(pop_size))
    if return_indices:
        return X_comb[sel], F_comb[sel], sel
    return X_comb[sel], F_comb[sel]


def _validate_reference_point(points: np.ndarray, reference_point: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference_point, dtype=np.float64)
    if ref.ndim != 1:
        raise ValueError("reference_point must be 1D.")
    if ref.shape[0] != points.shape[1]:
        raise ValueError("reference_point dimensionality mismatch.")
    if np.any(points > ref):
        ref = np.maximum(ref, points.max(axis=0) + 1e-9)
    return ref


def _hypervolume_2d(points: np.ndarray, ref: np.ndarray) -> float:
    if points.shape[0] == 0:
        return 0.0
    order = np.argsort(points[:, 0], kind="mergesort")
    sorted_points = points[order]
    widths = np.maximum(ref[0] - sorted_points[:, 0], 0.0)
    prev_f2 = np.minimum.accumulate(np.concatenate(([ref[1]], sorted_points[:-1, 1])))
    heights = np.maximum(prev_f2 - sorted_points[:, 1], 0.0)
    return float(np.sum(widths * heights))


def _hypervolume_3d(points: np.ndarray, ref: np.ndarray) -> float:
    order = np.argsort(points[:, 2], kind="mergesort")
    sorted_points = points[order]
    hv = 0.0
    prev_f3 = ref[2]
    for end in range(sorted_points.shape[0] - 1, -1, -1):
        f3 = sorted_points[end, 2]
        height = max(prev_f3 - f3, 0.0)
        if height <= 0.0:
            continue
        slab = _hypervolume_2d(sorted_points[: end + 1, :2], ref[:2])
        hv += slab * height
        prev_f3 = f3
    return float(hv)


def _hypervolume_recursive(points: np.ndarray, ref: np.ndarray) -> float:
    if points.size == 0:
        return 0.0
    n_obj = points.shape[1]
    if n_obj == 1:
        widths = np.maximum(ref[0] - points[:, 0], 0.0)
        return float(np.max(widths)) if widths.size else 0.0

    order = np.argsort(points[:, n_obj - 1], kind="mergesort")
    sorted_points = points[order]
    hv = 0.0
    bound = ref[n_obj - 1]
    while sorted_points.shape[0] > 0:
        current = sorted_points[-1, n_obj - 1]
        height = bound - current
        if height > 0.0:
            reduced = sorted_points[:, : n_obj - 1]
            hv += _hypervolume_recursive(reduced, ref[: n_obj - 1]) * height
            bound = current
        sorted_points = sorted_points[:-1]
    return float(hv)


def hypervolume(points: np.ndarray, reference_point: np.ndarray) -> float:
    pts = _require_float64_c("points", np.asarray(points), ndim=2)
    if pts.shape[0] == 0:
        return 0.0
    ref = _validate_reference_point(pts, reference_point)
    n_obj = pts.shape[1]
    if n_obj == 1:
        return float(np.max(np.maximum(ref[0] - pts[:, 0], 0.0)))
    if n_obj == 2:
        return _hypervolume_2d(pts, ref)
    if n_obj == 3:
        return _hypervolume_3d(pts, ref)
    return _hypervolume_recursive(pts, ref)


def _hypervolume_contributions_2d(points: np.ndarray, ref: np.ndarray) -> np.ndarray:
    unique_points, inverse, counts = np.unique(points, axis=0, return_inverse=True, return_counts=True)
    unique_contribs = np.zeros(unique_points.shape[0], dtype=np.float64)
    if unique_points.shape[0]:
        order = np.lexsort((unique_points[:, 1], unique_points[:, 0]))
        sorted_points = unique_points[order]
        prev_min = np.concatenate(([np.inf], np.minimum.accumulate(sorted_points[:-1, 1])))
        nd_sorted = sorted_points[:, 1] < prev_min
        nd_order = order[nd_sorted]
        if nd_order.size:
            nd_points = unique_points[nd_order]
            x = nd_points[:, 0]
            y = nd_points[:, 1]
            x_next = np.concatenate((x[1:], [ref[0]]))
            y_prev = np.concatenate(([ref[1]], y[:-1]))
            nd_contrib = np.maximum(x_next - x, 0.0) * np.maximum(y_prev - y, 0.0)
            unique_contribs[nd_order] = nd_contrib
    contrib = unique_contribs[inverse]
    contrib[counts[inverse] > 1] = 0.0
    return contrib


def _hypervolume_contributions_generic(points: np.ndarray, ref: np.ndarray) -> np.ndarray:
    n = points.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)
    full = hypervolume(points, ref)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        without_i = np.delete(points, i, axis=0)
        out[i] = full - hypervolume(without_i, ref)
    return out


def hypervolume_contributions(points: np.ndarray, reference_point: np.ndarray) -> np.ndarray:
    pts = _require_float64_c("points", np.asarray(points), ndim=2)
    if pts.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    ref = _validate_reference_point(pts, reference_point)
    if pts.shape[1] == 2:
        return _hypervolume_contributions_2d(pts, ref)
    return _hypervolume_contributions_generic(pts, ref)


def smsemoa_remove_index(F_combined: np.ndarray, reference_point: np.ndarray) -> int:
    F = _require_float64_c("F_combined", np.asarray(F_combined), ndim=2)
    ref = np.asarray(reference_point, dtype=np.float64)
    if F.shape[0] == 0:
        raise ValueError("F_combined cannot be empty.")
    ranks, _ = nsga2_ranking(F)
    worst_rank = int(ranks.max(initial=0))
    worst_idx = np.flatnonzero(ranks == worst_rank)
    if worst_idx.size == 1:
        return int(worst_idx[0])
    contribs = hypervolume_contributions(F[worst_idx], ref)
    return int(worst_idx[int(np.argmin(contribs))])


def dominance_matrix(F: np.ndarray) -> np.ndarray:
    F = _require_float64_c("F", np.asarray(F), ndim=2)
    n = F.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=bool)
    less_equal = F[:, None, :] <= F[None, :, :]
    strictly_less = F[:, None, :] < F[None, :, :]
    dom = np.logical_and(np.all(less_equal, axis=2), np.any(strictly_less, axis=2))
    np.fill_diagonal(dom, False)
    return dom


def spea2_fitness(F: np.ndarray, dom: np.ndarray | None = None, k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    F = _require_float64_c("F", np.asarray(F), ndim=2)
    n = F.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64), np.empty((0, 0), dtype=np.float64)
    dom_matrix = np.asarray(dom, dtype=bool) if dom is not None else dominance_matrix(F)
    if dom_matrix.shape != (n, n):
        raise ValueError("dom must have shape (N, N).")

    if k is None:
        k = max(1, int(np.sqrt(n)))
    k = min(int(k), n - 1) if n > 1 else 1

    strength = dom_matrix.sum(axis=1).astype(np.float64)
    raw_fitness = np.zeros(n, dtype=np.float64)
    for i in range(n):
        dominators = np.where(dom_matrix[:, i])[0]
        raw_fitness[i] = strength[dominators].sum()

    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(F[i] - F[j]))
            dist[i, j] = d
            dist[j, i] = d

    if n == 1:
        density = np.array([0.0], dtype=np.float64)
    else:
        density = np.zeros(n, dtype=np.float64)
        for i in range(n):
            sorted_d = np.sort(dist[i], kind="mergesort")
            sigma_k = sorted_d[k] if k < n else sorted_d[-1]
            density[i] = 1.0 / (sigma_k + 2.0)

    return raw_fitness + density, dist


def _truncate_by_distance_indices(dist: np.ndarray, keep: int, k: int) -> np.ndarray:
    candidates = list(range(dist.shape[0]))
    if len(candidates) <= keep:
        return np.asarray(candidates, dtype=np.int64)

    k = int(k)
    if k < 1:
        k = 1
    if len(candidates) > 1:
        k = min(k, len(candidates) - 1)

    while len(candidates) > keep:
        sub = dist[np.ix_(candidates, candidates)].copy()
        np.fill_diagonal(sub, np.inf)
        nearest = np.partition(sub, k, axis=1)[:, k]
        remove_pos = int(np.argmin(nearest))
        del candidates[remove_pos]

    return np.asarray(candidates, dtype=np.int64)


def spea2_environmental_selection_indices(F: np.ndarray, archive_size: int, k: int | None = None) -> np.ndarray:
    F = _require_float64_c("F", np.asarray(F), ndim=2)
    n = F.shape[0]
    keep = int(archive_size)
    if keep <= 0:
        return np.empty(0, dtype=np.int64)
    if n <= keep:
        return np.arange(n, dtype=np.int64)

    dom = dominance_matrix(F)
    strength = dom.sum(axis=1).astype(np.float64)
    raw = np.zeros(n, dtype=np.float64)
    for i in range(n):
        dominators = np.where(dom[:, i])[0]
        raw[i] = strength[dominators].sum()

    k_eff = int(k) if k is not None else 1
    if k_eff < 1:
        k_eff = 1

    unique_fitness = np.unique(raw)
    selected: list[int] = []
    for fit in np.sort(unique_fitness):
        front = np.flatnonzero(raw == fit)
        if len(selected) + front.size <= keep:
            selected.extend(front.tolist())
            continue

        remaining = keep - len(selected)
        if remaining <= 0:
            break
        front_F = F[front]
        dist_front = np.linalg.norm(front_F[:, None, :] - front_F[None, :, :], axis=2)
        local = _truncate_by_distance_indices(dist_front, remaining, k_eff)
        selected.extend(front[local].tolist())
        break

    return np.asarray(selected, dtype=np.int64)


def _epsilon_indicator(F: np.ndarray) -> np.ndarray:
    diff = F[None, :, :] - F[:, None, :]
    return np.asarray(np.max(diff, axis=2), dtype=np.float64)


def _hypervolume_indicator(F: np.ndarray, reference_point: np.ndarray | None = None) -> np.ndarray:
    n = F.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=np.float64)
    ref = np.asarray(reference_point, dtype=np.float64) if reference_point is not None else np.max(F, axis=0) + 1.0
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pair = np.vstack([F[i], F[j]])
            hv_pair = hypervolume(pair, ref)
            hv_j = hypervolume(F[j : j + 1], ref)
            out[i, j] = hv_j - hv_pair
    return out


def ibea_indicator_matrix(
    F: np.ndarray,
    reference_point: np.ndarray | None = None,
    kind: str = "epsilon",
) -> np.ndarray:
    F = _require_float64_c("F", np.asarray(F), ndim=2)
    key = str(kind).lower()
    if key == "hypervolume":
        return _hypervolume_indicator(F, reference_point)
    return _epsilon_indicator(F)


def ibea_environmental_selection_indices(
    F: np.ndarray,
    pop_size: int,
    reference_point: np.ndarray | None = None,
    kind: str = "epsilon",
    kappa: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    F = _require_float64_c("F", np.asarray(F), ndim=2)
    keep = int(pop_size)
    n = F.shape[0]
    if keep <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    if keep >= n:
        ind = ibea_indicator_matrix(F, reference_point, kind)
        mat = np.asarray(ind, dtype=np.float64)
        np.fill_diagonal(mat, np.inf)
        denom = float(kappa) if float(kappa) != 0.0 else 1.0e-12
        contrib = np.exp(-mat / denom)
        contrib[~np.isfinite(contrib)] = 0.0
        fitness = np.asarray(-np.sum(contrib, axis=1), dtype=np.float64)
        return np.arange(n, dtype=np.int64), fitness

    ind = np.asarray(ibea_indicator_matrix(F, reference_point, kind), dtype=np.float64)
    mat = ind.copy()
    np.fill_diagonal(mat, np.inf)
    denom = float(kappa) if float(kappa) != 0.0 else 1.0e-12
    contrib = np.exp(-mat / denom)
    contrib[~np.isfinite(contrib)] = 0.0
    fitness = np.asarray(-np.sum(contrib, axis=1), dtype=np.float64)
    selected = np.arange(n, dtype=np.int64)

    while selected.size > keep:
        worst = int(np.argmin(fitness))
        delta = np.exp(-ind[:, selected[worst]] / denom)
        delta[~np.isfinite(delta)] = 0.0
        fitness += delta[selected]
        fitness[worst] = fitness[worst] - delta[selected[worst]]
        selected = np.delete(selected, worst)
        fitness = np.delete(fitness, worst)

    return np.asarray(selected, dtype=np.int64), np.asarray(fitness, dtype=np.float64)


def sbx_crossover(
    X_parents: np.ndarray,
    prob: float,
    eta: float,
    xl: np.ndarray,
    xu: np.ndarray,
    seed: int,
    prob_var: float = 0.5,
) -> np.ndarray:
    parents = _require_float64_c("X_parents", np.asarray(X_parents), ndim=2)
    n_parents, n_var = parents.shape
    if n_parents == 0:
        return np.empty_like(parents)

    lower, upper = _normalize_bounds(np.asarray(xl), np.asarray(xu), n_var)
    rng = np.random.default_rng(_as_uint64_seed(seed))

    work = parents
    if n_parents % 2 != 0:
        work = np.vstack([parents, parents[-1:]])
        n_parents = work.shape[0]

    offspring = work.copy()
    eps = 1.0e-14
    for i in range(0, n_parents, 2):
        if rng.random() > prob:
            continue
        for j in range(n_var):
            if rng.random() > prob_var:
                continue

            y1 = float(offspring[i, j])
            y2 = float(offspring[i + 1, j])
            yl = float(lower[j])
            yu = float(upper[j])
            if abs(y1 - y2) < eps or yl >= yu:
                continue

            if y1 < y2:
                y1v, y2v = y1, y2
            else:
                y1v, y2v = y2, y1

            beta = 1.0 + (2.0 * (y1v - yl) / (y2v - y1v))
            alpha = 2.0 - beta ** -(eta + 1.0)
            rand = float(rng.random())
            if rand <= (1.0 / alpha):
                betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
            else:
                betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
            c1 = 0.5 * ((y1v + y2v) - betaq * (y2v - y1v))

            beta = 1.0 + (2.0 * (yu - y2v) / (y2v - y1v))
            alpha = 2.0 - beta ** -(eta + 1.0)
            if rand <= (1.0 / alpha):
                betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
            else:
                betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
            c2 = 0.5 * ((y1v + y2v) + betaq * (y2v - y1v))

            c1 = float(np.clip(c1, yl, yu))
            c2 = float(np.clip(c2, yl, yu))
            if rng.random() <= 0.5:
                offspring[i, j], offspring[i + 1, j] = c2, c1
            else:
                offspring[i, j], offspring[i + 1, j] = c1, c2
    return offspring


def polynomial_mutation(
    X: np.ndarray,
    prob: float,
    eta: float,
    xl: np.ndarray,
    xu: np.ndarray,
    seed: int,
    in_place: bool = False,
) -> np.ndarray:
    X_arr = _require_float64_c("X", np.asarray(X), ndim=2)
    out = X_arr if in_place else X_arr.copy()
    n_ind, n_var = out.shape
    if n_ind == 0:
        return out

    lower, upper = _normalize_bounds(np.asarray(xl), np.asarray(xu), n_var)
    rng = np.random.default_rng(_as_uint64_seed(seed))
    mut_pow = 1.0 / (eta + 1.0)

    for i in range(n_ind):
        for j in range(n_var):
            if rng.random() > prob:
                continue
            y = float(out[i, j])
            yl = float(lower[j])
            yu = float(upper[j])
            if yl >= yu:
                continue

            delta1 = (y - yl) / (yu - yl)
            delta2 = (yu - y) / (yu - yl)
            rnd = float(rng.random())

            if rnd <= 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (xy ** (eta + 1.0))
                deltaq = val**mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (xy ** (eta + 1.0))
                deltaq = 1.0 - val**mut_pow

            y = y + deltaq * (yu - yl)
            out[i, j] = float(np.clip(y, yl, yu))
    return out


def generate_offspring(
    X: np.ndarray,
    F: np.ndarray,
    n_offspring: int,
    xl: np.ndarray,
    xu: np.ndarray,
    config: Mapping[str, object],
    seed: int,
    out: np.ndarray | None = None,
) -> np.ndarray:
    X = _require_float64_c("X", np.asarray(X), ndim=2)
    F = _require_float64_c("F", np.asarray(F), ndim=2)
    n_offspring = int(n_offspring)
    if n_offspring <= 0:
        return np.empty((0, X.shape[1]), dtype=np.float64)

    rng = np.random.default_rng(_as_uint64_seed(seed))
    pressure = int(config.get("tournament_pressure", 2))
    sbx_prob = float(config.get("sbx_prob", 0.9))
    sbx_eta = float(config.get("sbx_eta", 20.0))
    pm_prob = float(config.get("pm_prob", 1.0 / max(1, X.shape[1])))
    pm_eta = float(config.get("pm_eta", 20.0))

    ranks, crowd = nsga2_ranking(F)
    parent_count = n_offspring if n_offspring % 2 == 0 else n_offspring + 1
    parent_idx = _tournament_selection_rng(ranks, crowd, pressure, rng, parent_count)
    parents = X[parent_idx]

    children = sbx_crossover(
        parents,
        prob=sbx_prob,
        eta=sbx_eta,
        xl=np.asarray(xl),
        xu=np.asarray(xu),
        seed=int(rng.integers(0, 2**63)),
    )
    children = children[:n_offspring].copy()
    children = polynomial_mutation(
        children,
        prob=pm_prob,
        eta=pm_eta,
        xl=np.asarray(xl),
        xu=np.asarray(xu),
        seed=int(rng.integers(0, 2**63)),
        in_place=False,
    )

    if out is not None:
        out_arr = _require_float64_c("out", np.asarray(out), ndim=2)
        if out_arr.shape != children.shape:
            raise ValueError("out has wrong shape.")
        out_arr[:] = children
        return out_arr
    return children


def smsemoa_generate_offspring(
    X: np.ndarray,
    F: np.ndarray,
    selection: str,
    pressure: int,
    xl: np.ndarray,
    xu: np.ndarray,
    config: Mapping[str, object],
    seed: int,
    out: np.ndarray | None = None,
) -> np.ndarray:
    X = _require_float64_c("X", np.asarray(X), ndim=2)
    F = _require_float64_c("F", np.asarray(F), ndim=2)
    if X.shape[0] == 0:
        raise ValueError("X cannot be empty.")

    rng = np.random.default_rng(_as_uint64_seed(seed))
    sel = str(selection).lower()
    if sel == "tournament":
        ranks, crowd = nsga2_ranking(F)
        parent_idx = _tournament_selection_rng(ranks, crowd, int(pressure), rng, 2)
    else:
        parent_idx = rng.choice(X.shape[0], size=2, replace=True).astype(np.int64)

    sbx_prob = float(config.get("sbx_prob", 0.9))
    sbx_eta = float(config.get("sbx_eta", 20.0))
    pm_prob = float(config.get("pm_prob", 1.0 / max(1, X.shape[1])))
    pm_eta = float(config.get("pm_eta", 20.0))

    parents = X[parent_idx]
    children = sbx_crossover(
        parents,
        prob=sbx_prob,
        eta=sbx_eta,
        xl=np.asarray(xl),
        xu=np.asarray(xu),
        seed=int(rng.integers(0, 2**63)),
    )
    child = children[:1].copy()
    child = polynomial_mutation(
        child,
        prob=pm_prob,
        eta=pm_eta,
        xl=np.asarray(xl),
        xu=np.asarray(xu),
        seed=int(rng.integers(0, 2**63)),
        in_place=False,
    )

    if out is not None:
        out_arr = _require_float64_c("out", np.asarray(out), ndim=2)
        if out_arr.shape != child.shape:
            raise ValueError("out has wrong shape.")
        out_arr[:] = child
        return out_arr
    return child


def spea2_generate_offspring(
    X: np.ndarray,
    F: np.ndarray,
    n_offspring: int,
    k_neighbors: int,
    xl: np.ndarray,
    xu: np.ndarray,
    config: Mapping[str, object],
    seed: int,
    out: np.ndarray | None = None,
) -> np.ndarray:
    X = _require_float64_c("X", np.asarray(X), ndim=2)
    F = _require_float64_c("F", np.asarray(F), ndim=2)
    n_offspring = int(n_offspring)
    if n_offspring <= 0:
        return np.empty((0, X.shape[1]), dtype=np.float64)

    rng = np.random.default_rng(_as_uint64_seed(seed))
    n = F.shape[0]
    if n == 0:
        raise ValueError("F cannot be empty.")

    dom = dominance_matrix(F)
    strength = dom.sum(axis=1).astype(np.float64)
    raw_fitness = np.zeros(n, dtype=np.float64)
    for i in range(n):
        dominators = np.where(dom[:, i])[0]
        raw_fitness[i] = strength[dominators].sum()

    dist = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=2)
    k_eff = int(k_neighbors)
    if k_eff < 1:
        k_eff = 1
    if n > 1:
        k_eff = min(k_eff, n - 1)
    else:
        k_eff = 1
    density = np.partition(dist, kth=k_eff, axis=1)[:, k_eff]

    # Raw fitness (lower is better), then density (higher is better), then random.
    ranks = np.argsort(np.argsort(raw_fitness, kind="mergesort"), kind="mergesort").astype(np.int64)
    parent_count = n_offspring * 2
    parent_idx = _tournament_selection_rng(ranks, density, 2, rng, parent_count)

    sbx_prob = float(config.get("sbx_prob", 0.9))
    sbx_eta = float(config.get("sbx_eta", 20.0))
    pm_prob = float(config.get("pm_prob", 1.0 / max(1, X.shape[1])))
    pm_eta = float(config.get("pm_eta", 20.0))

    parents = X[parent_idx]
    crossed = sbx_crossover(
        parents,
        prob=sbx_prob,
        eta=sbx_eta,
        xl=np.asarray(xl),
        xu=np.asarray(xu),
        seed=int(rng.integers(0, 2**63)),
    )
    offspring = crossed.reshape(n_offspring, 2, X.shape[1])[:, 0, :].copy()
    offspring = polynomial_mutation(
        offspring,
        prob=pm_prob,
        eta=pm_eta,
        xl=np.asarray(xl),
        xu=np.asarray(xu),
        seed=int(rng.integers(0, 2**63)),
        in_place=False,
    )

    if out is not None:
        out_arr = _require_float64_c("out", np.asarray(out), ndim=2)
        if out_arr.shape != offspring.shape:
            raise ValueError("out has wrong shape.")
        out_arr[:] = offspring
        return out_arr
    return offspring


def _extract_objectives(eval_result: object) -> np.ndarray:
    if hasattr(eval_result, "F"):
        return np.asarray(getattr(eval_result, "F"), dtype=np.float64)
    if isinstance(eval_result, dict):
        if "F" not in eval_result:
            raise ValueError("eval_fn returned dict without key 'F'.")
        return np.asarray(eval_result["F"], dtype=np.float64)
    return np.asarray(eval_result, dtype=np.float64)


def nsga2_evolve(
    X0: np.ndarray,
    F0: np.ndarray,
    xl: np.ndarray,
    xu: np.ndarray,
    config: Mapping[str, object],
    n_generations: int,
    seed: int,
    eval_fn: Callable[[np.ndarray], object],
) -> tuple[np.ndarray, np.ndarray]:
    X = _require_float64_c("X0", np.asarray(X0), ndim=2).copy()
    F = _require_float64_c("F0", np.asarray(F0), ndim=2).copy()
    n_generations = int(n_generations)
    rng = np.random.default_rng(_as_uint64_seed(seed))
    pop_size = X.shape[0]

    for _ in range(n_generations):
        X_off = generate_offspring(
            X,
            F,
            pop_size,
            np.asarray(xl),
            np.asarray(xu),
            config,
            seed=int(rng.integers(0, 2**63)),
        )
        eval_out = eval_fn(X_off)
        F_off = _extract_objectives(eval_out)
        X, F = nsga2_survival(X, F, X_off, F_off, pop_size)
    return X, F


def is_native_backend() -> bool:
    return False


def backend_info() -> dict[str, object]:
    return {"backend": "python-fallback", "native": False}


__all__ = [
    "backend_info",
    "crowding_distance",
    "dominance_matrix",
    "fast_non_dominated_sort",
    "generate_offspring",
    "hypervolume",
    "hypervolume_contributions",
    "ibea_environmental_selection_indices",
    "ibea_indicator_matrix",
    "is_native_backend",
    "nsga2_evolve",
    "nsga2_ranking",
    "nsga2_survival",
    "polynomial_mutation",
    "sbx_crossover",
    "smsemoa_generate_offspring",
    "smsemoa_remove_index",
    "spea2_environmental_selection_indices",
    "spea2_fitness",
    "spea2_generate_offspring",
    "tournament_selection",
]
