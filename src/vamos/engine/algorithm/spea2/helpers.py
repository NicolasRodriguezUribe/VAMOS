# algorithm/spea2/helpers.py
"""
Support functions for SPEA2.

This module contains the core SPEA2 selection functions:
- Dominance matrix computation
- SPEA2 fitness (strength + density)
- Environmental selection
- Distance-based truncation
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

from vamos.foundation.constraints.utils import compute_violation, is_feasible

if TYPE_CHECKING:
    pass


def dominance_matrix(F: np.ndarray, G: np.ndarray | None, constraint_mode: str) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Compute dominance matrix with optional feasibility-aware handling.

    Parameters
    ----------
    F : np.ndarray
        Objective values, shape (N, n_obj).
    G : np.ndarray | None
        Constraint values, shape (N, n_con) or None.
    constraint_mode : str
        Constraint handling mode: "none" or "feasibility".

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None, np.ndarray | None]
        (dominance_matrix, feasibility_mask, constraint_violations)
    """
    n = F.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=bool), None, None

    feas: np.ndarray | None = None
    cv: np.ndarray | None = None

    if constraint_mode and constraint_mode != "none" and G is not None:
        cv = compute_violation(G)
        feas = is_feasible(G)

    dom = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Feasibility-based dominance
            if feas is not None and cv is not None:
                if feas[i] and not feas[j]:
                    dom[i, j] = True
                    continue
                if not feas[i] and feas[j]:
                    continue
                if not feas[i] and not feas[j]:
                    if cv[i] < cv[j]:
                        dom[i, j] = True
                    continue
            # Standard Pareto dominance
            if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                dom[i, j] = True

    return dom, feas, cv


def spea2_fitness(F: np.ndarray, dom: np.ndarray, k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Compute SPEA2 fitness and distance matrix.

    SPEA2 fitness consists of:
    1. Raw fitness: sum of strengths of all dominators
    2. Density: based on k-th nearest neighbor distance

    Parameters
    ----------
    F : np.ndarray
        Objective values, shape (N, n_obj).
    dom : np.ndarray
        Dominance matrix, shape (N, N).
    k : int | None
        k for k-th nearest neighbor distance. Defaults to sqrt(N).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (fitness, distance_matrix)
    """
    n = F.shape[0]
    if n == 0:
        return np.empty(0), np.empty((0, 0))

    if k is None:
        k = max(1, int(np.sqrt(n)))
    k = min(k, n - 1) if n > 1 else 1

    # Strength: number of solutions each solution dominates
    strength = dom.sum(axis=1)

    # Raw fitness: sum of strengths of all dominators
    raw_fitness = np.zeros(n)
    for i in range(n):
        dominators = np.where(dom[:, i])[0]
        raw_fitness[i] = strength[dominators].sum()

    # Distance matrix in objective space
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(F[i] - F[j])
            dist[i, j] = d
            dist[j, i] = d

    # Density: based on k-th nearest neighbor distance
    if n == 1:
        density = np.array([0.0])
    else:
        density = np.zeros(n)
        for i in range(n):
            sorted_dists = np.sort(dist[i])
            sigma_k = sorted_dists[k] if k < n else sorted_dists[-1]
            density[i] = 1.0 / (sigma_k + 2.0)

    return raw_fitness + density, dist


def strength_raw_fitness(dom: np.ndarray) -> np.ndarray:
    """Compute SPEA2 raw fitness (strength ranking) from dominance matrix."""
    n = dom.shape[0]
    if n == 0:
        return np.empty(0, dtype=float)
    strength = dom.sum(axis=1)
    raw_fitness = np.zeros(n, dtype=float)
    for i in range(n):
        dominators = np.where(dom[:, i])[0]
        raw_fitness[i] = strength[dominators].sum()
    return raw_fitness


def knn_density(F: np.ndarray, k: int = 1) -> np.ndarray:
    """Compute k-th nearest neighbor distance for each solution (higher is better)."""
    n = F.shape[0]
    if n == 0:
        return np.empty(0, dtype=float)
    if n == 1:
        return np.full(1, np.inf, dtype=float)
    k = int(k)
    if k < 1:
        k = 1
    if k >= n:
        k = n - 1
    dist = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=2)
    kth = np.partition(dist, kth=k, axis=1)[:, k]
    return kth


def truncate_by_distance(dist_matrix: np.ndarray, keep: int) -> np.ndarray:
    """Truncate by iteratively removing solution with smallest distance.

    This is the SPEA2 truncation procedure that preserves boundary solutions.

    Parameters
    ----------
    dist_matrix : np.ndarray
        Distance matrix, shape (N, N).
    keep : int
        Number of solutions to keep.

    Returns
    -------
    np.ndarray
        Indices of retained solutions.
    """
    candidates = list(range(dist_matrix.shape[0]))
    if len(candidates) <= keep:
        return np.asarray(candidates, dtype=int)

    dist = dist_matrix.copy()
    while len(candidates) > keep:
        sub = dist[np.ix_(candidates, candidates)]
        np.fill_diagonal(sub, np.inf)
        nearest = np.partition(sub, 1, axis=1)[:, 1]
        remove_pos = int(np.argmin(nearest))
        del candidates[remove_pos]

    return np.asarray(candidates, dtype=int)


def environmental_selection(
    X: np.ndarray,
    F: np.ndarray,
    G: np.ndarray | None,
    archive_size: int,
    k_neighbors: int | None,
    constraint_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """SPEA2 environmental selection (sequential truncation).

    Select solutions for the next generation archive based on fitness.

    Parameters
    ----------
    X : np.ndarray
        Decision variables, shape (N, n_var).
    F : np.ndarray
        Objective values, shape (N, n_obj).
    G : np.ndarray | None
        Constraint values, shape (N, n_con) or None.
    archive_size : int
        Target archive size.
    k_neighbors : int | None
        k for density estimation.
    constraint_mode : str
        Constraint handling mode.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray | None]
        (X_selected, F_selected, G_selected)
    """
    n = F.shape[0]
    if n == 0:
        return X, F, G
    if n <= archive_size:
        return X, F, G

    dom, _, _ = dominance_matrix(F, G, constraint_mode)
    raw_fitness = strength_raw_fitness(dom)
    k = int(k_neighbors) if k_neighbors is not None else 1

    unique_fitness = np.unique(raw_fitness)
    fronts = [np.flatnonzero(raw_fitness == fit) for fit in np.sort(unique_fitness)]

    selected: list[int] = []
    for front in fronts:
        if len(selected) + front.size <= archive_size:
            selected.extend(front.tolist())
            continue

        remaining = archive_size - len(selected)
        if remaining <= 0:
            break

        # Sequential truncation within the splitting front (jMetalPy-style).
        keep = front.tolist()
        while len(keep) > remaining:
            density = knn_density(F[np.asarray(keep)], k)
            order = np.argsort(-density, kind="mergesort")
            keep = [keep[i] for i in order]
            keep.pop()
        selected.extend(keep)
        break

    selected_idx = np.asarray(selected, dtype=int)
    return X[selected_idx], F[selected_idx], G[selected_idx] if G is not None else None


def compute_selection_metrics(
    F: np.ndarray,
    G: np.ndarray | None,
    constraint_mode: str,
    kernel: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute selection metrics for mating selection.

    Parameters
    ----------
    F : np.ndarray
        Objective values.
    G : np.ndarray | None
        Constraint values.
    constraint_mode : str
        Constraint handling mode.
    kernel : KernelBackend
        Backend for ranking.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (ranks, crowding_distances)
    """
    if F.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)

    if constraint_mode and constraint_mode != "none" and G is not None:
        cv = compute_violation(G)
        feas = is_feasible(G)
        if feas.any():
            feas_idx = np.nonzero(feas)[0]
            ranks, crowd = kernel.nsga2_ranking(F[feas_idx])
            metrics_rank = np.full(F.shape[0], ranks.max(initial=0) + 1, dtype=int)
            metrics_crowd = np.zeros(F.shape[0], dtype=float)
            metrics_rank[feas_idx] = ranks
            metrics_crowd[feas_idx] = crowd
            metrics_crowd[~feas] = -cv[~feas]
            return metrics_rank, metrics_crowd
        ranks = np.zeros(F.shape[0], dtype=int)
        crowd = -cv
        return ranks, crowd

    ranks, crowd = kernel.nsga2_ranking(F)
    return np.asarray(ranks, dtype=int), np.asarray(crowd, dtype=float)


__all__ = [
    "dominance_matrix",
    "spea2_fitness",
    "strength_raw_fitness",
    "knn_density",
    "truncate_by_distance",
    "environmental_selection",
    "compute_selection_metrics",
]
