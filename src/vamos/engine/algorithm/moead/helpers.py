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

# Aggregation IDs for optional JIT path.
AGG_TCHEBYCHEFF = 0
AGG_WEIGHTED_SUM = 1
AGG_PBI = 2
AGG_MODIFIED_TCHEBYCHEFF = 3

_AGGREGATION_IDS: dict[str, int] = {
    "tchebycheff": AGG_TCHEBYCHEFF,
    "tchebychef": AGG_TCHEBYCHEFF,
    "tschebyscheff": AGG_TCHEBYCHEFF,
    "weighted_sum": AGG_WEIGHTED_SUM,
    "weightedsum": AGG_WEIGHTED_SUM,
    "penaltyboundaryintersection": AGG_PBI,
    "penalty_boundary_intersection": AGG_PBI,
    "pbi": AGG_PBI,
    "modifiedtchebycheff": AGG_MODIFIED_TCHEBYCHEFF,
    "modified_tchebycheff": AGG_MODIFIED_TCHEBYCHEFF,
}

_UPDATE_NEIGHBORHOOD_JIT: Callable[..., int] | None = None
_UPDATE_NEIGHBORHOOD_DISABLED = False
_DUMMY_G: np.ndarray | None = None
_DUMMY_CV: np.ndarray | None = None
_DUMMY_CHILD_G: np.ndarray | None = None


def _dummy_buffers() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    global _DUMMY_G, _DUMMY_CV, _DUMMY_CHILD_G
    if _DUMMY_G is None:
        _DUMMY_G = np.empty((0, 0), dtype=float)
    if _DUMMY_CV is None:
        _DUMMY_CV = np.empty(0, dtype=float)
    if _DUMMY_CHILD_G is None:
        _DUMMY_CHILD_G = np.empty(0, dtype=float)
    return _DUMMY_G, _DUMMY_CV, _DUMMY_CHILD_G


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


def resolve_aggregation_spec(name: str, params: Mapping[str, object]) -> tuple[int, float, float]:
    """Resolve aggregation ID and parameters for fast paths."""
    method = name.lower()
    agg_id = _AGGREGATION_IDS.get(method, -1)

    theta_raw = params.get("theta", 5.0)
    theta = float(theta_raw) if isinstance(theta_raw, (int, float, str)) else 5.0

    rho_raw = params.get("rho", 0.001)
    rho = float(rho_raw) if isinstance(rho_raw, (int, float, str)) else 0.001

    return agg_id, theta, rho


def _use_numba_moead() -> bool:
    import os

    flag = os.environ.get("VAMOS_USE_NUMBA_MOEAD")
    if flag is None or flag == "":
        return True
    return flag.lower() in {"1", "true", "yes", "on"}


def _get_update_neighborhood_numba() -> Callable[..., int] | None:
    global _UPDATE_NEIGHBORHOOD_JIT, _UPDATE_NEIGHBORHOOD_DISABLED
    if _UPDATE_NEIGHBORHOOD_DISABLED:
        return None
    if _UPDATE_NEIGHBORHOOD_JIT is not None:
        return _UPDATE_NEIGHBORHOOD_JIT
    if not _use_numba_moead():
        _UPDATE_NEIGHBORHOOD_DISABLED = True
        return None
    try:
        from numba import njit
    except ImportError:
        _UPDATE_NEIGHBORHOOD_DISABLED = True
        return None

    @njit(cache=True)  # type: ignore[untyped-decorator]
    def _update_neighborhood_numba(
        X: np.ndarray,
        F: np.ndarray,
        G: np.ndarray,
        cv: np.ndarray,
        weights: np.ndarray,
        weights_safe: np.ndarray,
        weights_unit: np.ndarray,
        ideal: np.ndarray,
        child: np.ndarray,
        child_f: np.ndarray,
        child_g: np.ndarray,
        child_cv: float,
        candidate_order: np.ndarray,
        replace_limit: int,
        agg_id: int,
        agg_theta: float,
        agg_rho: float,
        use_constraints: int,
    ) -> int:
        replacements = 0
        n_obj = ideal.shape[0]
        for idx in range(candidate_order.shape[0]):
            k = int(candidate_order[idx])

            # Compute aggregation for current and child.
            if agg_id == AGG_TCHEBYCHEFF:
                current_val = -1.0
                child_val = -1.0
                for j in range(n_obj):
                    diff_c = abs(F[k, j] - ideal[j])
                    diff_child = abs(child_f[j] - ideal[j])
                    w_eff = weights_safe[k, j]
                    val_c = w_eff * diff_c
                    val_child = w_eff * diff_child
                    if val_c > current_val:
                        current_val = val_c
                    if val_child > child_val:
                        child_val = val_child
            elif agg_id == AGG_WEIGHTED_SUM:
                current_val = 0.0
                child_val = 0.0
                for j in range(n_obj):
                    w = weights[k, j]
                    current_val += w * F[k, j]
                    child_val += w * child_f[j]
            elif agg_id == AGG_PBI:
                d1 = 0.0
                for j in range(n_obj):
                    d1 += (F[k, j] - ideal[j]) * weights_unit[k, j]
                d1 = abs(d1)

                d1_child = 0.0
                for j in range(n_obj):
                    d1_child += (child_f[j] - ideal[j]) * weights_unit[k, j]
                d1_child = abs(d1_child)

                d2 = 0.0
                d2_child = 0.0
                for j in range(n_obj):
                    w_unit = weights_unit[k, j]
                    diff_c = (F[k, j] - ideal[j]) - d1 * w_unit
                    diff_child = (child_f[j] - ideal[j]) - d1_child * w_unit
                    d2 += diff_c * diff_c
                    d2_child += diff_child * diff_child
                d2 = np.sqrt(d2)
                d2_child = np.sqrt(d2_child)
                current_val = d1 + agg_theta * d2
                child_val = d1_child + agg_theta * d2_child
            else:  # AGG_MODIFIED_TCHEBYCHEFF
                current_val = -1.0
                child_val = -1.0
                sum_c = 0.0
                sum_child = 0.0
                for j in range(n_obj):
                    diff_c = abs(F[k, j] - ideal[j])
                    diff_child = abs(child_f[j] - ideal[j])
                    w_eff = weights_safe[k, j]
                    val_c = w_eff * diff_c
                    val_child = w_eff * diff_child
                    if val_c > current_val:
                        current_val = val_c
                    if val_child > child_val:
                        child_val = val_child
                    sum_c += val_c
                    sum_child += val_child
                current_val = current_val + agg_rho * sum_c
                child_val = child_val + agg_rho * sum_child

            replace = False
            if use_constraints == 1:
                current_cv = cv[k]
                feas_child = child_cv <= 0.0
                feas_curr = current_cv <= 0.0
                if (not feas_curr) and feas_child:
                    replace = True
                elif feas_child and feas_curr:
                    replace = child_val < current_val
                else:
                    replace = child_cv < current_cv
            else:
                replace = child_val < current_val

            if not replace:
                continue

            X[k] = child
            F[k] = child_f
            if use_constraints == 1:
                G[k] = child_g
                cv[k] = child_cv
            replacements += 1
            if replacements >= replace_limit:
                break
        return replacements

    _UPDATE_NEIGHBORHOOD_JIT = _update_neighborhood_numba
    return _UPDATE_NEIGHBORHOOD_JIT


def _update_neighborhood_python(
    X: np.ndarray,
    F: np.ndarray,
    G: np.ndarray,
    cv: np.ndarray,
    weights: np.ndarray,
    weights_safe: np.ndarray,
    weights_unit: np.ndarray,
    ideal: np.ndarray,
    child: np.ndarray,
    child_f: np.ndarray,
    child_g: np.ndarray,
    child_cv: float,
    candidate_order: np.ndarray,
    replace_limit: int,
    agg_id: int,
    agg_theta: float,
    agg_rho: float,
    use_constraints: int,
) -> int:
    """Pure-Python fallback for neighborhood updates when numba is unavailable."""
    replacements = 0
    for idx in range(candidate_order.shape[0]):
        k = int(candidate_order[idx])

        diff_current = np.abs(F[k] - ideal)
        diff_child = np.abs(child_f - ideal)

        if agg_id == AGG_TCHEBYCHEFF:
            current_val = float(np.max(weights_safe[k] * diff_current))
            child_val = float(np.max(weights_safe[k] * diff_child))
        elif agg_id == AGG_WEIGHTED_SUM:
            current_val = float(np.dot(weights[k], F[k]))
            child_val = float(np.dot(weights[k], child_f))
        elif agg_id == AGG_PBI:
            d1 = abs(float(np.dot(F[k] - ideal, weights_unit[k])))
            d1_child = abs(float(np.dot(child_f - ideal, weights_unit[k])))
            d2 = float(np.linalg.norm((F[k] - ideal) - d1 * weights_unit[k]))
            d2_child = float(np.linalg.norm((child_f - ideal) - d1_child * weights_unit[k]))
            current_val = d1 + agg_theta * d2
            child_val = d1_child + agg_theta * d2_child
        else:  # AGG_MODIFIED_TCHEBYCHEFF
            weighted_current = weights_safe[k] * diff_current
            weighted_child = weights_safe[k] * diff_child
            current_val = float(np.max(weighted_current)) + agg_rho * float(np.sum(weighted_current))
            child_val = float(np.max(weighted_child)) + agg_rho * float(np.sum(weighted_child))

        replace = False
        if use_constraints == 1:
            current_cv = cv[k]
            feas_child = child_cv <= 0.0
            feas_curr = current_cv <= 0.0
            if (not feas_curr) and feas_child:
                replace = True
            elif feas_child and feas_curr:
                replace = child_val < current_val
            else:
                replace = child_cv < current_cv
        else:
            replace = child_val < current_val

        if not replace:
            continue

        X[k] = child
        F[k] = child_f
        if use_constraints == 1:
            G[k] = child_g
            cv[k] = child_cv
        replacements += 1
        if replacements >= replace_limit:
            break
    return replacements


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
    st: MOEADState,
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
    use_constraints = constraint_mode != "none"

    if candidate_order is None:
        candidate_order = st.neighbors[idx]
    if candidate_order.size == 0:
        return
    assert st.aggregator is not None
    child_cv = cv_penalty if constraint_mode != "none" else 0.0

    if use_constraints and st.cv is None and st.G is not None:
        st.cv = compute_violation(st.G)

    updater = _get_update_neighborhood_numba()
    if updater is None:
        updater = _update_neighborhood_python

    dummy_g, dummy_cv, dummy_child_g = _dummy_buffers()
    updater(
        st.X,
        st.F,
        st.G if st.G is not None else dummy_g,
        st.cv if st.cv is not None else dummy_cv,
        st.weights,
        st.weights_safe,
        st.weights_unit,
        st.ideal,
        child,
        child_f,
        child_g if child_g is not None else dummy_child_g,
        float(child_cv),
        candidate_order,
        int(st.replace_limit),
        int(st.aggregation_id),
        float(st.aggregation_theta),
        float(st.aggregation_rho),
        1 if use_constraints else 0,
    )


__all__ = [
    "tchebycheff",
    "weighted_sum",
    "pbi",
    "modified_tchebycheff",
    "build_aggregator",
    "resolve_aggregation_spec",
    "compute_neighbors",
    "update_neighborhood",
]
