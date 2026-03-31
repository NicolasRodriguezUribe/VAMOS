"""Tolerance-based row matching utilities for genealogy tracking."""

from __future__ import annotations

import numpy as np

_MATCH_TOLERANCE = 1e-12
_MATCH_COMPARISON_BUDGET = 2_000_000


def match_ids_by_tolerance(
    new_X: np.ndarray,
    combined_X: np.ndarray,
    combined_ids: np.ndarray,
    *,
    atol: float = _MATCH_TOLERANCE,
) -> np.ndarray:
    """Map rows in ``new_X`` back to ids from ``combined_X`` within a tolerance.

    Parameters
    ----------
    new_X : np.ndarray
        Surviving decision vectors after selection.
    combined_X : np.ndarray
        Combined parent + offspring decision matrix before selection.
    combined_ids : np.ndarray
        IDs aligned with ``combined_X``.
    atol : float, default ``1e-12``
        Absolute tolerance for row matching.

    Returns
    -------
    np.ndarray
        Matched ids for each row in ``new_X``. Unmatched rows receive ``-1``.
    """
    new_arr = np.asarray(new_X, dtype=float)
    combined_arr = np.asarray(combined_X, dtype=float)
    ids_arr = np.asarray(combined_ids, dtype=int)

    if new_arr.ndim != 2 or combined_arr.ndim != 2:
        raise ValueError("new_X and combined_X must be 2D arrays.")
    if new_arr.shape[1] != combined_arr.shape[1]:
        raise ValueError("new_X and combined_X must have the same number of columns.")
    if ids_arr.shape[0] != combined_arr.shape[0]:
        raise ValueError("combined_ids must align with combined_X rows.")

    new_ids = np.full(new_arr.shape[0], -1, dtype=int)
    if new_arr.shape[0] == 0 or combined_arr.shape[0] == 0:
        return new_ids

    unresolved = np.ones(new_arr.shape[0], dtype=bool)
    n_var = max(1, combined_arr.shape[1])

    start = 0
    while start < combined_arr.shape[0] and np.any(unresolved):
        unresolved_idx = np.flatnonzero(unresolved)
        comparisons_per_row = max(1, unresolved_idx.size * n_var)
        block_size = max(1, _MATCH_COMPARISON_BUDGET // comparisons_per_row)
        stop = min(combined_arr.shape[0], start + block_size)

        block = combined_arr[start:stop]
        pending_rows = new_arr[unresolved_idx]
        matches = np.all(np.abs(pending_rows[:, None, :] - block[None, :, :]) <= atol, axis=2)
        has_match = np.any(matches, axis=1)
        if np.any(has_match):
            matched_rows = unresolved_idx[has_match]
            first_match_local = np.argmax(matches[has_match], axis=1)
            new_ids[matched_rows] = ids_arr[start + first_match_local]
            unresolved[matched_rows] = False

        start = stop

    return new_ids


__all__ = ["match_ids_by_tolerance"]
