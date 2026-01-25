"""SMS-EMOA helper functions.

This module contains utility functions for SMS-EMOA:
- Reference point initialization and management
- Survival selection using hypervolume contributions
- Population/offspring evaluation with constraints
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from vamos.foundation.metrics.hypervolume import hypervolume_contributions

if TYPE_CHECKING:
    from .state import SMSEMOAState
    from vamos.foundation.kernel.backend import KernelBackend
    from vamos.foundation.problem.types import ProblemProtocol


__all__ = [
    "initialize_reference_point",
    "update_reference_point",
    "survival_selection",
    "evaluate_population_with_constraints",
]


def _is_duplicate_row(matrix: np.ndarray, row: np.ndarray) -> bool:
    if matrix.size == 0:
        return False
    return bool(np.any(np.all(matrix == row, axis=1)))


def initialize_reference_point(
    F: np.ndarray,
    ref_cfg: Mapping[str, object],
) -> tuple[np.ndarray, float, bool]:
    """Initialize reference point for HV computation.

    Parameters
    ----------
    F : np.ndarray
        Initial population objective values, shape (pop_size, n_obj).
    ref_cfg : dict
        Reference point configuration with keys:
        - offset (float): Offset from nadir point (default 1.0)
        - adaptive (bool): Whether to adaptively update (default True)
        - vector (list): Explicit reference point vector (optional)

    Returns
    -------
    tuple
        (ref_point, ref_offset, ref_adaptive)
    """
    offset_raw = ref_cfg.get("offset", 1.0)
    offset = float(offset_raw) if isinstance(offset_raw, (int, float, str)) else 1.0
    adaptive = bool(ref_cfg.get("adaptive", True))
    vector = ref_cfg.get("vector")

    if vector is not None:
        ref = np.asarray(vector, dtype=float)
        if ref.shape[0] != F.shape[1]:
            raise ValueError("reference_point vector dimensionality mismatch.")
        ref = np.maximum(ref, F.max(axis=0) + offset)
    else:
        ref = F.max(axis=0) + offset

    return ref, offset, adaptive


def update_reference_point(
    ref_point: np.ndarray,
    F_new: np.ndarray,
    ref_offset: float,
) -> np.ndarray:
    """Update reference point from current objective values.

    Parameters
    ----------
    ref_point : np.ndarray
        Current reference point.
    F_new : np.ndarray
        Current objective values, shape (n, n_obj).
    ref_offset : float
        Offset to add beyond observed maximum.

    Returns
    -------
    np.ndarray
        Updated reference point.
    """
    return np.asarray(F_new.max(axis=0) + ref_offset, dtype=float)


def survival_selection(
    st: SMSEMOAState,
    X_child: np.ndarray,
    F_child: np.ndarray,
    G_child: np.ndarray | None,
    kernel: KernelBackend,
) -> None:
    """Perform survival selection, removing worst HV contributor.

    SMS-EMOA survival selection combines the population with offspring,
    performs non-dominated ranking, and removes the solution with the
    smallest hypervolume contribution from the worst rank.

    Parameters
    ----------
    st : SMSEMOAState
        Algorithm state to update.
    X_child : np.ndarray
        Child decision vectors, shape (1, n_var).
    F_child : np.ndarray
        Child objective values, shape (1, n_obj).
    G_child : np.ndarray | None
        Child constraint values if any.
    kernel : KernelBackend
        Backend for ranking operations.
    """
    pop_n = int(st.X.shape[0])
    if pop_n == 0:
        raise ValueError("Cannot perform survival selection with an empty population.")

    if X_child.shape[0] != 1 or F_child.shape[0] != 1:
        raise ValueError("survival_selection expects exactly one offspring (shape (1, ...)).")

    if st.eliminate_duplicates:
        child_x = X_child[0]
        child_f = F_child[0]
        if _is_duplicate_row(st.X, child_x) or _is_duplicate_row(st.F, child_f):
            return

    # Build combined objective matrix in a reusable buffer to avoid per-iteration allocations.
    F_work = st._survival_F
    if F_work is None or F_work.shape != (pop_n + 1, st.F.shape[1]):
        F_work = np.empty((pop_n + 1, st.F.shape[1]), dtype=st.F.dtype)
        st._survival_F = F_work
    F_work[:pop_n] = st.F
    F_work[pop_n] = F_child[0]

    # Update reference point if adaptive (kept consistent with the previous implementation:
    # computed on the combined (pop + child) set, before removal).
    if st.ref_adaptive:
        st.ref_point = update_reference_point(st.ref_point, F_work, st.ref_offset)

    # Non-dominated ranking (on combined set)
    ranks, _ = kernel.nsga2_ranking(F_work)
    worst_rank = ranks.max()
    worst_idx = np.flatnonzero(ranks == worst_rank)

    if worst_idx.size == 1:
        remove_idx = int(worst_idx[0])
    else:
        contribs = hypervolume_contributions(F_work[worst_idx], st.ref_point)
        remove_idx = int(worst_idx[int(np.argmin(contribs))])

    # If the offspring is removed, population remains unchanged.
    if remove_idx == pop_n:
        return

    # Remove an existing individual and append the child at the end, preserving order.
    if remove_idx < pop_n - 1:
        st.X[remove_idx : pop_n - 1] = st.X[remove_idx + 1 : pop_n]
        st.F[remove_idx : pop_n - 1] = st.F[remove_idx + 1 : pop_n]
        if st.G is not None and G_child is not None:
            st.G[remove_idx : pop_n - 1] = st.G[remove_idx + 1 : pop_n]
        if st.ids is not None and st.pending_offspring_ids is not None:
            st.ids[remove_idx : pop_n - 1] = st.ids[remove_idx + 1 : pop_n]

    st.X[pop_n - 1] = X_child[0]
    st.F[pop_n - 1] = F_child[0]
    if st.G is not None and G_child is not None:
        st.G[pop_n - 1] = G_child[0]
    if st.ids is not None and st.pending_offspring_ids is not None:
        st.ids[pop_n - 1] = st.pending_offspring_ids[0]


def evaluate_population_with_constraints(
    problem: ProblemProtocol,
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
        (F, G) where F is objective values and G is constraint values (or None).
    """
    n_obj = problem.n_obj
    n_constr = getattr(problem, "n_constr", getattr(problem, "n_con", 0)) or 0

    out: dict[str, np.ndarray] = {"F": np.empty((X.shape[0], n_obj), dtype=np.float64)}
    if n_constr > 0:
        out["G"] = np.empty((X.shape[0], n_constr), dtype=np.float64)

    # Support both in-place and "replace out['F']" problem implementations.
    problem.evaluate(X, out)
    return out["F"], out.get("G")
