"""SMS-EMOA helper functions.

This module contains utility functions for SMS-EMOA:
- Reference point initialization and management
- Survival selection using hypervolume contributions
- Population/offspring evaluation with constraints
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vamos.engine.algorithm.components.hypervolume import hypervolume_contributions

if TYPE_CHECKING:
    from .state import SMSEMOAState
    from vamos.foundation.kernel.protocols import KernelBackend
    from vamos.foundation.problem.protocol import ProblemProtocol


__all__ = [
    "initialize_reference_point",
    "update_reference_point",
    "survival_selection",
    "evaluate_population_with_constraints",
]


def initialize_reference_point(
    F: np.ndarray,
    ref_cfg: dict,
) -> tuple[np.ndarray, float, bool]:
    """Initialize reference point for HV computation.

    Parameters
    ----------
    F : np.ndarray
        Initial population objective values, shape (pop_size, n_obj).
    ref_cfg : dict
        Reference point configuration with keys:
        - offset (float): Offset from nadir point (default 0.1)
        - adaptive (bool): Whether to adaptively update (default True)
        - vector (list): Explicit reference point vector (optional)

    Returns
    -------
    tuple
        (ref_point, ref_offset, ref_adaptive)
    """
    offset = float(ref_cfg.get("offset", 0.1))
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
    """Update reference point with new objective values.

    Parameters
    ----------
    ref_point : np.ndarray
        Current reference point.
    F_new : np.ndarray
        New objective values, shape (n_new, n_obj).
    ref_offset : float
        Offset to add beyond observed maximum.

    Returns
    -------
    np.ndarray
        Updated reference point.
    """
    return np.maximum(ref_point, F_new.max(axis=0) + ref_offset)


def survival_selection(
    st: "SMSEMOAState",
    X_child: np.ndarray,
    F_child: np.ndarray,
    G_child: np.ndarray | None,
    kernel: "KernelBackend",
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
    # Combine population with child
    X_comb = np.vstack([st.X, X_child])
    F_comb = np.vstack([st.F, F_child])
    G_comb = (
        np.vstack([st.G, G_child])
        if st.G is not None and G_child is not None
        else None
    )

    # Combine ids if genealogy is enabled
    ids_comb = None
    if st.ids is not None and st.pending_offspring_ids is not None:
        ids_comb = np.concatenate([st.ids, st.pending_offspring_ids])

    # Update reference point if adaptive
    if st.ref_adaptive:
        st.ref_point = update_reference_point(st.ref_point, F_child, st.ref_offset)

    # Non-dominated ranking
    ranks, _ = kernel.nsga2_ranking(F_comb)
    worst_rank = ranks.max()
    worst_idx = np.flatnonzero(ranks == worst_rank)

    if worst_idx.size == 1:
        remove_idx = worst_idx[0]
    else:
        # Remove solution with smallest HV contribution
        contribs = hypervolume_contributions(F_comb[worst_idx], st.ref_point)
        remove_idx = worst_idx[np.argmin(contribs)]

    # Keep all except removed
    keep = np.delete(np.arange(F_comb.shape[0]), remove_idx)
    st.X = X_comb[keep][: st.pop_size]
    st.F = F_comb[keep][: st.pop_size]
    if G_comb is not None:
        st.G = G_comb[keep][: st.pop_size]
    if ids_comb is not None:
        st.ids = ids_comb[keep][: st.pop_size]


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
        (F, G) where F is objective values and G is constraint values (or None).
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
