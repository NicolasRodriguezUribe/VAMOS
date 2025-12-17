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
    from vamos.foundation.problem.protocol import ProblemProtocol


__all__ = [
    "fast_non_dominated_sort",
    "identify_extremes",
    "compute_intercepts",
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
    S = [[] for _ in range(n)]
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
    """Identify extreme points using ASF (Achievement Scalarization Function).

    Parameters
    ----------
    shifted : np.ndarray
        Shifted objective values (F - ideal), shape (n, n_obj).

    Returns
    -------
    np.ndarray
        Indices of extreme points for each objective.
    """
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
    """Compute intercepts from extreme points.

    Falls back to axis-wise maxima if plane solving fails.

    Parameters
    ----------
    shifted : np.ndarray
        Shifted objective values.
    extreme_idx : np.ndarray
        Indices of extreme points.

    Returns
    -------
    np.ndarray
        Intercepts for normalization.
    """
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
    """Perform niche-based selection from critical front.

    Selects solutions to maintain diversity across reference directions.

    Parameters
    ----------
    front : np.ndarray
        Indices of solutions in the critical front.
    n_remaining : int
        Number of solutions to select.
    niche_counts : np.ndarray
        Current count of solutions per reference direction.
    associations : np.ndarray
        Reference direction association for each solution.
    distances : np.ndarray
        Perpendicular distance to associated reference direction.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Indices of selected solutions.
    """
    selected: list[int] = []
    pool = front.tolist()
    while len(selected) < n_remaining and pool:
        assoc_front = np.array([associations[idx] for idx in pool])
        unique_refs = np.unique(assoc_front)
        ref_counts = niche_counts[unique_refs]
        min_count = np.min(ref_counts)
        candidate_refs = unique_refs[ref_counts == min_count]
        ref_choice = rng.choice(candidate_refs)

        candidates = [idx for idx in pool if associations[idx] == ref_choice]
        if not candidates:
            niche_counts[ref_choice] = np.inf
            continue
        cand_dist = np.array([distances[idx] for idx in candidates])
        best = candidates[int(np.argmin(cand_dist))]
        pool.remove(best)
        niche_counts[ref_choice] += 1
        selected.append(best)

    if len(selected) < n_remaining and pool:
        remaining = n_remaining - len(selected)
        additional = rng.choice(pool, size=min(remaining, len(pool)), replace=False)
        selected.extend(additional.tolist())

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
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

    ideal = F_all.min(axis=0)
    shifted = F_all - ideal
    extreme_idx = identify_extremes(shifted)
    intercepts = compute_intercepts(shifted, extreme_idx)
    denom = np.where(intercepts > 0, intercepts, 1.0)
    normalized = shifted / denom

    associations, distances = associate(normalized, ref_dirs_norm)
    niche_counts = np.zeros(ref_dirs_norm.shape[0], dtype=int)

    for front in fronts:
        front_arr = np.asarray(front, dtype=int)
        if len(survivor_indices) + front_arr.size <= pop_size:
            survivor_indices.extend(front)
            if new_G is not None and G_all is not None:
                new_G.extend(G_all[front_arr])
            for idx in front:
                niche_counts[associations[idx]] += 1
        else:
            remaining = pop_size - len(survivor_indices)
            selected_idx = niche_selection(
                front_arr, remaining, niche_counts, associations, distances, rng
            )
            survivor_indices.extend(selected_idx.tolist())
            if new_G is not None and G_all is not None:
                new_G.extend(G_all[selected_idx])
            break

    survivor_arr = np.asarray(survivor_indices, dtype=int)
    return (
        X_all[survivor_arr],
        F_all[survivor_arr],
        np.asarray(new_G) if new_G is not None else None,
        survivor_arr,
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

    F = np.empty((X.shape[0], n_obj), dtype=np.float64)

    if n_con > 0:
        G = np.empty((X.shape[0], n_con), dtype=np.float64)
        problem.evaluate(X, {"F": F, "G": G})
    else:
        problem.evaluate(X, {"F": F})
        G = None

    return F, G
