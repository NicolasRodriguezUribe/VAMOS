# algorithm/ibea/helpers.py
"""
Support functions for IBEA.

This module contains the core IBEA selection functions:
- Indicator computation (epsilon, hypervolume)
- Fitness calculation
- Environmental selection
- Constraint handling
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vamos.foundation.constraints.utils import compute_violation, is_feasible
from vamos.foundation.metrics.hypervolume import hypervolume

if TYPE_CHECKING:
    pass


def epsilon_indicator(F: np.ndarray) -> np.ndarray:
    """Compute additive epsilon indicator matrix.

    Parameters
    ----------
    F : np.ndarray
        Objective values, shape (N, n_obj).

    Returns
    -------
    np.ndarray
        Epsilon indicator matrix, shape (N, N).
    """
    # Match jMetalPy: epsilon(i, j) = max_k (f_jk - f_ik)
    diff = F[None, :, :] - F[:, None, :]
    return np.asarray(np.max(diff, axis=2), dtype=float)


def hypervolume_indicator(F: np.ndarray) -> np.ndarray:
    """Compute hypervolume indicator matrix.

    Parameters
    ----------
    F : np.ndarray
        Objective values, shape (N, n_obj).

    Returns
    -------
    np.ndarray
        Hypervolume indicator matrix, shape (N, N).
    """
    n = F.shape[0]
    if n == 0:
        return np.empty((0, 0))
    ref = np.max(F, axis=0) + 1.0
    indicator = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pair = np.vstack([F[i], F[j]])
            hv_pair = hypervolume(pair, ref)
            hv_j = hypervolume(F[j : j + 1], ref)
            indicator[i, j] = hv_j - hv_pair
    return indicator


def compute_indicator_matrix(F: np.ndarray, indicator: str) -> np.ndarray:
    """Compute indicator matrix based on selected type.

    Parameters
    ----------
    F : np.ndarray
        Objective values.
    indicator : str
        Indicator type: "epsilon" or "hypervolume".

    Returns
    -------
    np.ndarray
        Indicator matrix.
    """
    if indicator == "hypervolume":
        return hypervolume_indicator(F)
    return epsilon_indicator(F)


def ibea_fitness(indicator: np.ndarray, kappa: float) -> np.ndarray:
    """Compute IBEA fitness from indicator matrix.

    IBEA fitness is the negative sum of exponential indicator contributions.
    Lower fitness values are worse (more negative); the worst is removed.

    Parameters
    ----------
    indicator : np.ndarray
        Indicator matrix, shape (N, N).
    kappa : float
        Scaling factor controlling selection pressure.

    Returns
    -------
    np.ndarray
        Fitness values, shape (N,).
    """
    mat = indicator.copy()
    np.fill_diagonal(mat, np.inf)
    contrib = np.exp(-mat / kappa)
    contrib[~np.isfinite(contrib)] = 0.0
    return np.asarray(-np.sum(contrib, axis=1), dtype=float)


def apply_constraint_penalty(fitness: np.ndarray, G: np.ndarray | None) -> np.ndarray:
    """Apply constraint penalty to fitness values.

    Parameters
    ----------
    fitness : np.ndarray
        Fitness values.
    G : np.ndarray | None
        Constraint values.

    Returns
    -------
    np.ndarray
        Penalized fitness values.
    """
    if G is None:
        return np.asarray(fitness, dtype=float)
    cv = compute_violation(G)
    feas = is_feasible(G)
    if not feas.any():
        return np.asarray(fitness + cv, dtype=float)
    penalty = np.max(np.abs(fitness)) + 1.0
    penalized = fitness.copy()
    penalized[~feas] += penalty * (1.0 + cv[~feas])
    return penalized


def environmental_selection(
    X: np.ndarray,
    F: np.ndarray,
    G: np.ndarray | None,
    pop_size: int,
    indicator: str,
    kappa: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Perform IBEA environmental selection.

    Iteratively removes the worst individual until population size is reached.

    Parameters
    ----------
    X : np.ndarray
        Decision variables, shape (N, n_var).
    F : np.ndarray
        Objective values, shape (N, n_obj).
    G : np.ndarray | None
        Constraint values or None.
    pop_size : int
        Target population size.
    indicator : str
        Indicator type.
    kappa : float
        Scaling factor.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]
        (X_selected, F_selected, G_selected, fitness)
    """
    ind = compute_indicator_matrix(F, indicator)
    fitness = ibea_fitness(ind, kappa)
    fitness = apply_constraint_penalty(fitness, G)

    while X.shape[0] > pop_size:
        worst = int(np.argmin(fitness))
        delta = np.exp(-ind[:, worst] / kappa)
        delta[worst] = 0.0
        fitness += delta
        X = np.delete(X, worst, axis=0)
        F = np.delete(F, worst, axis=0)
        if G is not None:
            G = np.delete(G, worst, axis=0)
        ind = np.delete(np.delete(ind, worst, axis=0), worst, axis=1)
        fitness = np.delete(fitness, worst, axis=0)
    return X, F, G, fitness


def combine_constraints(G: np.ndarray | None, G_off: np.ndarray | None) -> np.ndarray | None:
    """Combine parent and offspring constraints.

    Parameters
    ----------
    G : np.ndarray | None
        Parent constraints.
    G_off : np.ndarray | None
        Offspring constraints.

    Returns
    -------
    np.ndarray | None
        Combined constraints.
    """
    if G is None and G_off is None:
        return None
    if G is None:
        return G_off
    if G_off is None:
        return G
    return np.vstack([G, G_off])


__all__ = [
    "epsilon_indicator",
    "hypervolume_indicator",
    "compute_indicator_matrix",
    "ibea_fitness",
    "apply_constraint_penalty",
    "environmental_selection",
    "combine_constraints",
]
