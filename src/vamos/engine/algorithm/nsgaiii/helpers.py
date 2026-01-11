"""NSGA-III helper functions.

This module provides helper functions for NSGA-III including:
- Non-dominated sorting
- Reference point association
- Niching-based survival selection
- ASF (Achievement Scalarization Function) for extremes
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vamos.foundation.problem.types import ProblemProtocol


__all__ = [
    "fast_non_dominated_sort",
    "identify_extremes",
    "compute_intercepts",
    "get_extreme_points",
    "get_nadir_point",
    "associate",
    "niche_selection",
    "nsgaiii_survival",
    "evaluate_population_with_constraints",
]


def fast_non_dominated_sort(F: np.ndarray) -> list[list[int]]:
    """Fast non-dominated sorting.

    Parameters
    ----------
    F : np.ndarray
        Objective values, shape (n, n_obj).

    Returns
    -------
    list[list[int]]
        List of fronts, where each front is a list of indices.
    """
    n = F.shape[0]
    S: list[list[int]] = [[] for _ in range(n)]
    domination_count = np.zeros(n, dtype=int)
    ranks = np.zeros(n, dtype=int)
    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                S[p].append(q)
            elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            ranks[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front: list[int] = []
        for p in fronts[i]:
            for q in S[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    ranks[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    fronts.pop()  # remove last empty front
    return fronts


def identify_extremes(shifted: np.ndarray) -> np.ndarray:
    """Identify extreme points using ASF (Achievement Scalarization Function)."""
    if shifted.size == 0:
        return np.array([], dtype=int)
    n_obj = shifted.shape[1]
    extremes = np.empty(n_obj, dtype=int)
    unit = np.eye(n_obj)
    for i in range(n_obj):
        weights = np.where(unit[i] == 0, 1e6, 1.0)
        asf = (shifted * weights).max(axis=1)
        extremes[i] = int(np.argmin(asf))
    return extremes


def compute_intercepts(shifted: np.ndarray, extreme_idx: np.ndarray) -> np.ndarray:
    """Compute intercepts from extreme points (legacy helper)."""
    n_obj = shifted.shape[1]
    if extreme_idx.size == 0:
        return np.ones(n_obj, dtype=float)
    extreme_pts = shifted[extreme_idx]
    intercepts = np.zeros(n_obj, dtype=float)
    try:
        b = np.ones(n_obj)
        plane = np.linalg.solve(extreme_pts, b)
        intercepts = 1.0 / plane
    except Exception:
        intercepts = shifted.max(axis=0)
    if np.any(~np.isfinite(intercepts)) or np.any(intercepts <= 1e-12):
        intercepts = shifted.max(axis=0)
    intercepts = np.where(intercepts > 0, intercepts, 1.0)
    return intercepts


def get_extreme_points(
    F: np.ndarray,
    n_obj: int,
    ideal_point: np.ndarray,
    extreme_points: np.ndarray | None = None,
) -> np.ndarray:
    """Identify extreme points using ASF, preserving previous extremes."""
    if F.size == 0:
        return np.empty((0, n_obj), dtype=float)
    base = F
    if extreme_points is not None and extreme_points.size:
        base = np.vstack([extreme_points, F])

    shifted = base - ideal_point
    shifted[shifted < 1e-3] = 0.0

    asf = np.eye(n_obj)
    asf[asf == 0] = 1e6
    vals = np.max(shifted * asf[:, None, :], axis=2)
    idx = np.argmin(vals, axis=1)
    return np.asarray(base[idx], dtype=float)


def get_nadir_point(
    extreme_points: np.ndarray,
    ideal_point: np.ndarray,
    worst_point: np.ndarray,
    worst_of_front: np.ndarray,
    worst_of_population: np.ndarray,
) -> np.ndarray:
    """Compute nadir point using extreme points (Deb & Jain 2014)."""
    try:
        M = extreme_points - ideal_point
        b = np.ones(extreme_points.shape[1])
        plane = np.linalg.solve(M, b)
        intercepts = 1.0 / plane
        nadir = ideal_point + intercepts
        if not np.allclose(M @ plane, b) or np.any(intercepts <= 1e-6) or np.any(nadir > worst_point):
            raise np.linalg.LinAlgError
    except Exception:
        nadir = worst_of_front

    mask = nadir - ideal_point <= 1e-6
    nadir = nadir.copy()
    nadir[mask] = worst_of_population[mask]
    return nadir


def associate(
    normalized_F: np.ndarray,
    ref_dirs_norm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Associate solutions with reference directions.

    Parameters
    ----------
    normalized_F : np.ndarray
        Normalized objective values, shape (n, n_obj).
    ref_dirs_norm : np.ndarray
        Normalized reference directions, shape (n_ref, n_obj).

    Returns
    -------
    tuple
        (associations, distances) where associations are ref direction indices
        and distances are perpendicular distances to the directions.
    """
    norms = np.linalg.norm(normalized_F, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1e-12)
    normalized_vectors = normalized_F / norms
    cosine = normalized_vectors @ ref_dirs_norm.T
    cosine = np.clip(cosine, -1.0, 1.0)
    associations = np.argmax(cosine, axis=1)
    cos_selected = cosine[np.arange(cosine.shape[0]), associations]
    distances = norms.flatten() * np.sqrt(1.0 - np.square(cos_selected))
    return associations, distances


def niche_selection(
    front: np.ndarray,
    n_remaining: int,
    niche_counts: np.ndarray,
    associations: np.ndarray,
    distances: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Perform niche-based selection from the critical front (NSGA-III)."""
    selected: list[int] = []
    if front.size == 0 or n_remaining <= 0:
        return np.empty(0, dtype=int)

    mask = np.ones(front.size, dtype=bool)
    while len(selected) < n_remaining:
        remaining = n_remaining - len(selected)
        active_pos = np.where(mask)[0]
        if active_pos.size == 0:
            break

        assoc_front = associations[front[active_pos]]
        unique_refs = np.unique(assoc_front)
        ref_counts = niche_counts[unique_refs]
        min_count = np.min(ref_counts)
        candidate_refs = unique_refs[ref_counts == min_count]
        rng.shuffle(candidate_refs)
        candidate_refs = candidate_refs[:remaining]

        for ref_choice in candidate_refs:
            candidates = active_pos[assoc_front == ref_choice]
            if candidates.size == 0:
                niche_counts[ref_choice] = np.inf
                continue
            rng.shuffle(candidates)
            if niche_counts[ref_choice] == 0:
                cand_dist = distances[front[candidates]]
                pick = candidates[int(np.argmin(cand_dist))]
            else:
                pick = candidates[0]
            mask[pick] = False
            selected.append(int(front[pick]))
            niche_counts[ref_choice] += 1
            if len(selected) >= n_remaining:
                break

    return np.asarray(selected, dtype=int)


def nsgaiii_survival(
    X: np.ndarray,
    F: np.ndarray,
    G: np.ndarray | None,
    X_off: np.ndarray,
    F_off: np.ndarray,
    G_off: np.ndarray | None,
    pop_size: int,
    ref_dirs_norm: np.ndarray,
    rng: np.random.Generator,
    ideal_point: np.ndarray,
    extreme_points: np.ndarray | None,
    worst_point: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray,
]:
    """Perform NSGA-III survival selection with niching.

    Parameters
    ----------
    X : np.ndarray
        Parent decision vectors.
    F : np.ndarray
        Parent objective values.
    G : np.ndarray or None
        Parent constraint values.
    X_off : np.ndarray
        Offspring decision vectors.
    F_off : np.ndarray
        Offspring objective values.
    G_off : np.ndarray or None
        Offspring constraint values.
    pop_size : int
        Target population size.
    ref_dirs_norm : np.ndarray
        Normalized reference directions.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    tuple
        (X_new, F_new, G_new, survivor_indices) where survivor_indices
        are indices into the combined [X, X_off] array.
    """
    X_all = np.vstack([X, X_off])
    F_all = np.vstack([F, F_off])
    G_all = np.vstack([G, G_off]) if G is not None and G_off is not None else None

    fronts = fast_non_dominated_sort(F_all)
    survivor_indices: list[int] = []
    new_G: list[np.ndarray] | None = [] if G_all is not None else None

    ideal_point = np.minimum(ideal_point, F_all.min(axis=0))
    worst_point = np.maximum(worst_point, F_all.max(axis=0))

    if fronts:
        front0 = np.asarray(fronts[0], dtype=int)
        nd_F = F_all[front0]
    else:
        nd_F = F_all

    extreme_points = get_extreme_points(nd_F, F_all.shape[1], ideal_point, extreme_points)
    worst_of_population = F_all.max(axis=0)
    worst_of_front = nd_F.max(axis=0) if nd_F.size else worst_of_population
    nadir_point = get_nadir_point(extreme_points, ideal_point, worst_point, worst_of_front, worst_of_population)

    denom = nadir_point - ideal_point
    denom = np.where(denom == 0, 1e-12, denom)
    normalized = (F_all - ideal_point) / denom

    associations, distances = associate(normalized, ref_dirs_norm)
    niche_counts = np.zeros(ref_dirs_norm.shape[0], dtype=int)

    last_front: np.ndarray | None = None
    for front in fronts:
        front_arr = np.asarray(front, dtype=int)
        if len(survivor_indices) + front_arr.size <= pop_size:
            survivor_indices.extend(front)
            if new_G is not None and G_all is not None:
                new_G.extend(G_all[front_arr])
            for idx in front:
                niche_counts[associations[idx]] += 1
        else:
            last_front = front_arr
            break

    if last_front is not None and len(survivor_indices) < pop_size:
        remaining = pop_size - len(survivor_indices)
        selected_idx = niche_selection(last_front, remaining, niche_counts, associations, distances, rng)
        survivor_indices.extend(selected_idx.tolist())
        if new_G is not None and G_all is not None:
            new_G.extend(G_all[selected_idx])

    survivor_arr = np.asarray(survivor_indices, dtype=int)
    return (
        X_all[survivor_arr],
        F_all[survivor_arr],
        np.asarray(new_G) if new_G is not None else None,
        survivor_arr,
        ideal_point,
        extreme_points,
        worst_point,
    )


def evaluate_population_with_constraints(
    problem: "ProblemProtocol",
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Evaluate population and compute constraints if present.

    Parameters
    ----------
    problem : ProblemProtocol
        Problem to evaluate.
    X : np.ndarray
        Decision vectors, shape (pop_size, n_var).

    Returns
    -------
    tuple
        (F, G) objective and constraint values.
    """
    n_obj = problem.n_obj
    n_con = getattr(problem, "n_con", 0) or 0

    out: dict[str, np.ndarray] = {"F": np.empty((X.shape[0], n_obj), dtype=np.float64)}
    if n_con > 0:
        out["G"] = np.empty((X.shape[0], n_con), dtype=np.float64)

    # Some problems write into preallocated arrays (out["F"][:]=...),
    # while others replace out["F"] entirely (out["F"]=...). Always read back
    # from the out dict to support both.
    problem.evaluate(X, out)
    return out["F"], out.get("G")
