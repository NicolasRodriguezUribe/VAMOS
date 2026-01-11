"""SMPSO helper functions.

This module contains utility functions for SMPSO:
- Repair strategy resolution
- Dominance comparison for personal best updates
- Personal best update logic with constraint handling
- Evaluation result extraction
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vamos.operators.impl.real import (
    ClampRepair,
    ReflectRepair,
    ResampleRepair,
    RoundRepair,
)
from vamos.foundation.constraints.utils import compute_violation, is_feasible


__all__ = [
    "REPAIR_MAP",
    "resolve_repair",
    "dominates",
    "update_personal_bests",
    "extract_eval_arrays",
]


REPAIR_MAP: dict[str, type] = {
    "clip": ClampRepair,
    "clamp": ClampRepair,
    "reflect": ReflectRepair,
    "random": ResampleRepair,
    "resample": ResampleRepair,
    "round": RoundRepair,
}


def resolve_repair(cfg: Any | None) -> Any | None:
    """Resolve repair strategy from configuration.

    Parameters
    ----------
    cfg : Any or None
        Repair configuration. Can be:
        - None: No repair
        - tuple: (method, params)
        - dict: {"method": ..., **params}
        - str: Method name

    Returns
    -------
    Any or None
        Repair operator instance or None.

    Raises
    ------
    ValueError
        If repair strategy is unknown.
    """
    if cfg is None:
        return None
    if isinstance(cfg, tuple):
        method, params = cfg
        params = dict(params)
    elif isinstance(cfg, dict):
        cfg = dict(cfg)
        method = cfg.pop("method", cfg.pop("name", cfg.pop("type", None)))
        params = cfg
    else:
        method, params = str(cfg), {}

    if method is None:
        return None
    normalized = str(method).lower()
    if normalized in {"none", "off", "disabled"}:
        return None
    cls = REPAIR_MAP.get(normalized)
    if cls is None:
        raise ValueError(f"Unknown repair strategy '{method}' for SMPSO.")
    return cls(**params)


def dominates(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Check Pareto dominance (a dominates b).

    Parameters
    ----------
    a : np.ndarray
        Objective values, shape (n, n_obj).
    b : np.ndarray
        Objective values, shape (n, n_obj).

    Returns
    -------
    np.ndarray
        Boolean array where True means a[i] dominates b[i].
    """
    leq = a <= b
    lt = a < b
    return np.asarray(np.all(leq, axis=1) & np.any(lt, axis=1), dtype=bool)


def update_personal_bests(
    X: np.ndarray,
    F: np.ndarray,
    G: np.ndarray | None,
    pbest_X: np.ndarray,
    pbest_F: np.ndarray,
    pbest_G: np.ndarray | None,
    constraint_mode: str,
) -> None:
    """Update personal best positions based on dominance.

    For constrained problems, uses feasibility-first approach:
    1. Feasible always beats infeasible
    2. Among feasible, dominance applies
    3. Among infeasible, lower constraint violation wins

    Parameters
    ----------
    X : np.ndarray
        Current positions.
    F : np.ndarray
        Current objective values.
    G : np.ndarray or None
        Current constraint values.
    pbest_X : np.ndarray
        Personal best positions (updated in-place).
    pbest_F : np.ndarray
        Personal best objectives (updated in-place).
    pbest_G : np.ndarray or None
        Personal best constraints (updated in-place).
    constraint_mode : str
        Constraint handling mode.
    """
    if constraint_mode and constraint_mode != "none" and G is not None and pbest_G is not None:
        feas_new = is_feasible(G)
        feas_old = is_feasible(pbest_G)
        cv_new = compute_violation(G)
        cv_old = compute_violation(pbest_G)

        better = np.zeros(X.shape[0], dtype=bool)
        # Feasible beats infeasible
        better |= feas_new & ~feas_old
        # Among feasible, dominance wins
        better |= feas_new & feas_old & dominates(F, pbest_F)
        # Among infeasible, lower violation wins
        better |= (~feas_new) & (~feas_old) & (cv_new < cv_old)
        update_idx = better
    else:
        # Without constraints, update if current is not dominated by personal best
        update_idx = ~dominates(pbest_F, F)

    pbest_X[update_idx] = X[update_idx]
    pbest_F[update_idx] = F[update_idx]
    if pbest_G is not None and G is not None:
        pbest_G[update_idx] = G[update_idx]


def extract_eval_arrays(eval_result: Any) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract F and G arrays from evaluation result.

    Parameters
    ----------
    eval_result : Any
        Evaluation result. Can be:
        - Object with .F and optional .G attributes
        - dict with "F" and optional "G" keys
        - ndarray (treated as F only)

    Returns
    -------
    tuple
        (F, G) objective and constraint arrays.
    """
    if hasattr(eval_result, "F"):
        F_val = getattr(eval_result, "F", None)
        if F_val is None:
            raise ValueError("Evaluation result is missing F.")
        G_val = getattr(eval_result, "G", None)
        return np.asarray(F_val, dtype=float), np.asarray(G_val, dtype=float) if G_val is not None else None
    if isinstance(eval_result, dict):
        F_val = eval_result.get("F")
        if F_val is None:
            raise ValueError("Evaluation result dict is missing 'F'.")
        G_val = eval_result.get("G")
        return np.asarray(F_val, dtype=float), np.asarray(G_val, dtype=float) if G_val is not None else None
    return np.asarray(eval_result, dtype=float), None
