"""
Result assembly helpers for engine algorithms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from vamos.engine.algorithm.components.state import AlgorithmState


def build_result(
    state: AlgorithmState,
    hv_reached: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the result dictionary from algorithm state.

    Parameters
    ----------
    state : AlgorithmState
        Current algorithm state.
    hv_reached : bool
        Whether HV threshold was reached.
    extra : dict[str, Any] | None
        Additional result fields.

    Returns
    -------
    dict[str, Any]
        Result dictionary with X, F, evaluations, and optional archive.
    """
    result: dict[str, Any] = {
        "X": state.X,
        "F": state.F,
        "evaluations": state.n_eval,
        "hv_reached": hv_reached,
    }

    if state.G is not None:
        result["G"] = state.G

    archive_payload = get_external_archive_payload(state)
    if archive_payload is not None:
        result["archive"] = archive_payload

    if extra:
        result.update(extra)

    return result


def _archive_payload_from_holder(holder: Any) -> dict[str, Any] | None:
    if holder is None:
        return None

    contents_payload = getattr(holder, "contents_payload", None)
    if callable(contents_payload):
        payload = contents_payload()
        if isinstance(payload, dict):
            return dict(payload)

    contents_with_constraints = getattr(holder, "contents_with_constraints", None)
    if callable(contents_with_constraints):
        archive_X, archive_F, archive_G = contents_with_constraints()
        return {
            "X": archive_X,
            "F": archive_F,
            "G": archive_G,
            "size": int(archive_F.shape[0]),
        }

    contents = getattr(holder, "contents", None)
    if callable(contents):
        archive_X, archive_F = contents()
        return {
            "X": archive_X,
            "F": archive_F,
            "size": int(archive_F.shape[0]),
        }

    return None


def get_external_archive_payload(state: Any) -> dict[str, Any] | None:
    """Return external archive payload regardless of the concrete archive holder."""
    result_archive = getattr(state, "result_archive", None)
    archive_payload = _archive_payload_from_holder(result_archive)
    if archive_payload is not None:
        return archive_payload

    archive_manager = getattr(state, "archive_manager", None)
    archive_payload = _archive_payload_from_holder(archive_manager)
    if archive_payload is not None:
        return archive_payload

    archive = getattr(state, "archive", None)
    if archive is not None:
        archive_X = getattr(archive, "X", None)
        archive_F = getattr(archive, "F", None)
        if archive_X is not None and archive_F is not None:
            archive_G = getattr(archive, "G", None)
            return {
                "X": archive_X,
                "F": archive_F,
                "G": archive_G,
                "size": int(np.asarray(archive_F).shape[0]),
            }

    archive_X = getattr(state, "archive_X", None)
    archive_F = getattr(state, "archive_F", None)
    if archive_X is not None and archive_F is not None:
        archive_G = getattr(state, "archive_G", None)
        return {
            "X": archive_X,
            "F": archive_F,
            "G": archive_G,
            "size": int(np.asarray(archive_F).shape[0]),
        }
    return None


def get_external_archive_contents(state: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Return external archive contents regardless of the concrete archive holder."""
    archive_payload = get_external_archive_payload(state)
    if archive_payload is None:
        return None
    return np.asarray(archive_payload["X"]), np.asarray(archive_payload["F"])


def wants_population_result(state: Any) -> bool:
    mode = str(getattr(state, "result_mode", "non_dominated") or "non_dominated").strip().lower()
    return mode == "population"


__all__ = ["build_result", "get_external_archive_contents", "get_external_archive_payload", "wants_population_result"]
