"""
Archive management helpers for algorithms.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

from vamos.engine.algorithm.components.archive import (
    CrowdingDistanceArchive,
    HypervolumeArchive,
)

if TYPE_CHECKING:
    from vamos.engine.algorithm.components.state import AlgorithmState
    from vamos.foundation.kernel.backend import KernelBackend


def setup_archive(
    kernel: "KernelBackend",
    X: np.ndarray,
    F: np.ndarray,
    n_var: int,
    n_obj: int,
    dtype: np.dtype,
    archive_size: int | None,
    archive_type: str = "crowding",
) -> tuple[np.ndarray | None, np.ndarray | None, CrowdingDistanceArchive | HypervolumeArchive | None]:
    """
    Initialize external archive if configured.

    Parameters
    ----------
    kernel : KernelBackend
        The kernel backend.
    X : np.ndarray
        Initial population decision variables.
    F : np.ndarray
        Initial population objectives.
    n_var : int
        Number of decision variables.
    n_obj : int
        Number of objectives.
    dtype : np.dtype
        Data type for arrays.
    archive_size : int | None
        Archive capacity, or None to disable.
    archive_type : str
        Archive type: "crowding" or "hypervolume".

    Returns
    -------
    tuple
        (archive_X, archive_F, archive_manager)
    """
    if not archive_size:
        return None, None, None

    if archive_type == "hypervolume":
        archive_manager = HypervolumeArchive(archive_size, n_var, n_obj, dtype)
    else:
        archive_manager = CrowdingDistanceArchive(archive_size, n_var, n_obj, dtype)

    archive_X, archive_F = archive_manager.update(X, F)
    return archive_X, archive_F, archive_manager


def update_archive(
    state: "AlgorithmState",
    X_new: np.ndarray | None = None,
    F_new: np.ndarray | None = None,
) -> None:
    """
    Update archive with new solutions.

    Parameters
    ----------
    state : AlgorithmState
        Algorithm state with archive fields.
    X_new : np.ndarray | None
        New decision variables to add (defaults to state.X).
    F_new : np.ndarray | None
        New objectives to add (defaults to state.F).
    """
    if state.archive_manager is None:
        return

    X = X_new if X_new is not None else state.X
    F = F_new if F_new is not None else state.F

    state.archive_X, state.archive_F = state.archive_manager.update(X, F)


def resolve_archive_size(cfg: dict[str, Any]) -> int | None:
    """
    Resolve archive size from configuration.

    Parameters
    ----------
    cfg : dict[str, Any]
        Algorithm configuration.

    Returns
    -------
    int | None
        Archive size if configured and positive, else None.
    """
    archive_cfg = cfg.get("archive") or cfg.get("external_archive")
    if not archive_cfg:
        return None
    size = int(archive_cfg.get("size", 0))
    return size if size > 0 else None


__all__ = ["setup_archive", "update_archive", "resolve_archive_size"]
