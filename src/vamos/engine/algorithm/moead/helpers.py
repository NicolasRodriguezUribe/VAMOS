# algorithm/moead/helpers.py
"""
Support functions for MOEA/D.

This module contains neighborhood update logic and scalarization functions
(aggregation methods) for decomposition-based optimization.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, TypeAlias

import numpy as np

from vamos.foundation.constraints.utils import compute_violation

if TYPE_CHECKING:
    from .state import MOEADState

AggregatorFn: TypeAlias = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


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
    weighted = np.where(weights == 0, 0.0001 * diff, weights * diff)
    return np.asarray(np.max(weighted, axis=-1), dtype=float)


def weighted_sum(fvals: np.ndarray, weights: np.ndarray, ideal: np.ndarray) -> np.ndarray:
    """Weighted sum aggregation: sum(w * f).

    Parameters
    ----------
    fvals : np.ndarray
        Objective values.
    weights : np.ndarray
        Weight vectors.
    ideal : np.ndarray
        Ideal point (unused; kept for a consistent signature).

    Returns
    -------
    np.ndarray
        Aggregated scalar values.
    """
    return np.asarray(np.sum(weights * fvals, axis=-1), dtype=float)


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
    return np.asarray(d1 + theta * d2, dtype=float)


def modified_tchebycheff(fvals: np.ndarray, weights: np.ndarray, ideal: np.ndarray, rho: float = 0.001) -> np.ndarray:
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
    weighted = np.where(weights == 0, 0.0001 * diff, weights * diff)
    return np.asarray(np.max(weighted, axis=-1) + rho * np.sum(weighted, axis=-1), dtype=float)


def build_aggregator(name: str, params: Mapping[str, object]) -> AggregatorFn:
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
        theta_raw = params.get("theta", 5.0)
        theta = float(theta_raw) if isinstance(theta_raw, (int, float, str)) else 5.0

        def _agg(fvals: np.ndarray, weights: np.ndarray, ideal: np.ndarray) -> np.ndarray:
            return pbi(fvals, weights, ideal, theta)

        return _agg
    if method in {"modifiedtchebycheff", "modified_tchebycheff"}:
        rho_raw = params.get("rho", 0.001)
        rho = float(rho_raw) if isinstance(rho_raw, (int, float, str)) else 0.001

        def _agg(fvals: np.ndarray, weights: np.ndarray, ideal: np.ndarray) -> np.ndarray:
            return modified_tchebycheff(fvals, weights, ideal, rho)

        return _agg
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
    candidate_order: np.ndarray | None = None,
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
    candidate_order : np.ndarray | None
        Optional ordered candidate indices for replacement. When None, uses
        the precomputed neighborhood order.
    """
    constraint_mode = st.constraint_mode
    if constraint_mode == "none" or st.G is None or child_g is None:
        constraint_mode = "none"

    if candidate_order is None:
        candidate_order = st.neighbors[idx]
    if candidate_order.size == 0:
        return
    assert st.aggregator is not None
    replacements = 0
    child_cv = cv_penalty if constraint_mode != "none" else 0.0

    for k in candidate_order:
        weight = st.weights[k]
        current_val = float(np.asarray(st.aggregator(st.F[k], weight, st.ideal)).reshape(-1)[0])
        child_val = float(np.asarray(st.aggregator(child_f, weight, st.ideal)).reshape(-1)[0])

        replace = False
        if constraint_mode != "none":
            current_cv = compute_violation(st.G[k : k + 1])[0] if st.G is not None else 0.0
            feas_child = child_cv <= 0.0
            feas_curr = current_cv <= 0.0
            if not feas_curr and feas_child:
                replace = True
            elif feas_child and feas_curr:
                replace = child_val < current_val
            elif (not feas_child) and (not feas_curr):
                replace = child_cv < current_cv
        else:
            replace = child_val < current_val

        if not replace:
            continue

        st.X[k] = child
        st.F[k] = child_f
        if st.G is not None and child_g is not None:
            st.G[k] = child_g
        replacements += 1
        if replacements >= st.replace_limit:
            break


__all__ = [
    "tchebycheff",
    "weighted_sum",
    "pbi",
    "modified_tchebycheff",
    "build_aggregator",
    "compute_neighbors",
    "update_neighborhood",
]
