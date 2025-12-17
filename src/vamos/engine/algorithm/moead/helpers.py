# algorithm/moead/helpers.py
"""
Support functions for MOEA/D.

This module contains neighborhood update logic and scalarization functions
(aggregation methods) for decomposition-based optimization.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from vamos.foundation.constraints.utils import compute_violation

if TYPE_CHECKING:
    from .state import MOEADState


# =============================================================================
# Aggregation / Scalarization Functions
# =============================================================================

def tchebycheff(fvals: np.ndarray, weights: np.ndarray, ideal: np.ndarray) -> np.ndarray:
    """Tchebycheff aggregation: max(w * |f - z*|).

    Parameters
    ----------
    fvals : np.ndarray
        Objective values, shape (N, n_obj) or (n_obj,).
    weights : np.ndarray
        Weight vectors, shape (N, n_obj) or (n_obj,).
    ideal : np.ndarray
        Ideal point (minimum objectives seen), shape (n_obj,).

    Returns
    -------
    np.ndarray
        Aggregated scalar values, shape (N,) or scalar.
    """
    diff = np.abs(fvals - ideal)
    return np.max(weights * diff, axis=-1)


def weighted_sum(fvals: np.ndarray, weights: np.ndarray, ideal: np.ndarray) -> np.ndarray:
    """Weighted sum aggregation: sum(w * (f - z*)).

    Parameters
    ----------
    fvals : np.ndarray
        Objective values.
    weights : np.ndarray
        Weight vectors.
    ideal : np.ndarray
        Ideal point.

    Returns
    -------
    np.ndarray
        Aggregated scalar values.
    """
    shifted = fvals - ideal
    return np.sum(weights * shifted, axis=-1)


def pbi(fvals: np.ndarray, weights: np.ndarray, ideal: np.ndarray, theta: float = 5.0) -> np.ndarray:
    """Penalty boundary intersection (PBI) aggregation.

    Parameters
    ----------
    fvals : np.ndarray
        Objective values.
    weights : np.ndarray
        Weight vectors.
    ideal : np.ndarray
        Ideal point.
    theta : float
        Penalty parameter (default 5.0).

    Returns
    -------
    np.ndarray
        Aggregated scalar values.
    """
    diff = fvals - ideal
    norm_w = np.linalg.norm(weights, axis=-1, keepdims=True)
    norm_w = np.where(norm_w > 0, norm_w, 1.0)
    w_unit = weights / norm_w
    d1 = np.abs(np.sum(diff * w_unit, axis=-1))
    proj = (d1[..., None]) * w_unit
    d2 = np.linalg.norm(diff - proj, axis=-1)
    return d1 + theta * d2


def modified_tchebycheff(
    fvals: np.ndarray, weights: np.ndarray, ideal: np.ndarray, rho: float = 0.001
) -> np.ndarray:
    """Modified Tchebycheff: max component plus weighted L1 term.

    Parameters
    ----------
    fvals : np.ndarray
        Objective values.
    weights : np.ndarray
        Weight vectors.
    ideal : np.ndarray
        Ideal point.
    rho : float
        L1 penalty factor (default 0.001).

    Returns
    -------
    np.ndarray
        Aggregated scalar values.
    """
    diff = np.abs(fvals - ideal)
    weighted = weights * diff
    return np.max(weighted, axis=-1) + rho * np.sum(weighted, axis=-1)


def build_aggregator(name: str, params: dict) -> Callable:
    """Build aggregation function from name and parameters.

    Parameters
    ----------
    name : str
        Aggregation method name: "tchebycheff", "weighted_sum", "pbi", or
        "modified_tchebycheff".
    params : dict
        Additional parameters for the aggregation method.

    Returns
    -------
    Callable
        Aggregation function with signature (fvals, weights, ideal) -> values.

    Raises
    ------
    ValueError
        If the aggregation method is not supported.
    """
    method = name.lower()
    if method in {"tchebycheff", "tchebychef", "tschebyscheff"}:
        return tchebycheff
    if method in {"weighted_sum", "weightedsum"}:
        return weighted_sum
    if method in {"penaltyboundaryintersection", "penalty_boundary_intersection", "pbi"}:
        theta = float(params.get("theta", 5.0))
        return lambda fvals, weights, ideal: pbi(fvals, weights, ideal, theta)
    if method in {"modifiedtchebycheff", "modified_tchebycheff"}:
        rho = float(params.get("rho", 0.001))
        return lambda fvals, weights, ideal: modified_tchebycheff(fvals, weights, ideal, rho)
    raise ValueError(f"Unsupported aggregation method '{name}'.")


# =============================================================================
# Neighborhood Management
# =============================================================================

def compute_neighbors(weights: np.ndarray, neighbor_size: int) -> np.ndarray:
    """Compute neighborhood indices based on weight vector distances.

    Parameters
    ----------
    weights : np.ndarray
        Weight vectors, shape (pop_size, n_obj).
    neighbor_size : int
        Neighborhood size (T parameter).

    Returns
    -------
    np.ndarray
        Neighborhood indices, shape (pop_size, neighbor_size).
    """
    dist = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=2)
    order = np.argsort(dist, axis=1)
    return order[:, :neighbor_size]


def update_neighborhood(
    st: "MOEADState",
    idx: int,
    child: np.ndarray,
    child_f: np.ndarray,
    child_g: np.ndarray | None,
    cv_penalty: float,
) -> None:
    """Update neighborhood with a new offspring using aggregation comparison.

    This implements the MOEA/D replacement strategy where an offspring can
    replace at most `replace_limit` neighboring solutions if it improves
    their aggregated fitness.

    Parameters
    ----------
    st : MOEADState
        Algorithm state.
    idx : int
        Index of the current subproblem.
    child : np.ndarray
        Offspring decision variables.
    child_f : np.ndarray
        Offspring objective values.
    child_g : np.ndarray | None
        Offspring constraint values.
    cv_penalty : float
        Offspring constraint violation penalty.
    """
    constraint_mode = st.constraint_mode
    if constraint_mode == "none" or st.G is None or child_g is None:
        constraint_mode = "none"

    neighbor_idx = st.neighbors[idx]
    if neighbor_idx.size == 0:
        return

    local_weights = st.weights[neighbor_idx]
    current_vals = st.aggregator(st.F[neighbor_idx], local_weights, st.ideal)
    child_vals = st.aggregator(child_f.reshape(1, -1), local_weights, st.ideal).ravel()

    if constraint_mode != "none":
        child_cv = cv_penalty
        current_cv = compute_violation(st.G[neighbor_idx]) if st.G is not None else np.zeros_like(current_vals)
        feas_child = child_cv <= 0.0
        feas_curr = current_cv <= 0.0

        better_mask = np.zeros_like(current_vals, dtype=bool)
        better_mask |= (~feas_curr & feas_child)
        if feas_child:
            better_mask |= (feas_curr & (child_vals < current_vals))
        else:
            better_mask |= (~feas_curr & (child_cv < current_cv))

        if not np.any(better_mask):
            return
        candidates = neighbor_idx[better_mask]
    else:
        improved_mask = child_vals < current_vals
        if not np.any(improved_mask):
            return
        candidates = neighbor_idx[improved_mask]

    if candidates.size > st.replace_limit:
        replace_idx = st.rng.choice(candidates.size, size=st.replace_limit, replace=False)
        candidates = candidates[replace_idx]

    st.X[candidates] = child
    st.F[candidates] = child_f
    if st.G is not None and child_g is not None:
        st.G[candidates] = child_g


__all__ = [
    "tchebycheff",
    "weighted_sum",
    "pbi",
    "modified_tchebycheff",
    "build_aggregator",
    "compute_neighbors",
    "update_neighborhood",
]
