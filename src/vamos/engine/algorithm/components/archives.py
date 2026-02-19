"""
Archive management helpers for algorithms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.archive import ExternalArchiveConfig
from vamos.engine.algorithm.components.archive import (
    CrowdingDistanceArchive,
    HypervolumeArchive,
    UnboundedArchive,
)

if TYPE_CHECKING:
    from vamos.engine.algorithm.components.state import AlgorithmState
    from vamos.foundation.kernel.backend import KernelBackend

_ArchiveManager = CrowdingDistanceArchive | HypervolumeArchive | UnboundedArchive


def resolve_external_archive(cfg: dict[str, Any]) -> ExternalArchiveConfig | None:
    """Extract :class:`ExternalArchiveConfig` from a serialised config dict.

    The config dict is typically produced by ``AlgorithmConfig.to_dict()``.

    Returns ``None`` when no external archive is configured.
    """
    raw = cfg.get("external_archive")
    if raw is None:
        return None
    if isinstance(raw, ExternalArchiveConfig):
        return raw
    # Deserialise from plain dict (produced by dataclasses.asdict)
    return ExternalArchiveConfig(**raw)


def setup_archive(
    kernel: KernelBackend,
    X: np.ndarray,
    F: np.ndarray,
    n_var: int,
    n_obj: int,
    dtype: np.dtype,
    ext_cfg: ExternalArchiveConfig | None,
) -> tuple[np.ndarray | None, np.ndarray | None, _ArchiveManager | None]:
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
    ext_cfg : ExternalArchiveConfig | None
        External archive configuration, or ``None`` to disable.

    Returns
    -------
    tuple
        (archive_X, archive_F, archive_manager)
    """
    if ext_cfg is None:
        return None, None, None

    capacity = ext_cfg.capacity
    pruning = ext_cfg.pruning

    tol = ext_cfg.objective_tolerance

    if capacity is None:
        # Unbounded archive
        manager: _ArchiveManager = UnboundedArchive(n_var=n_var, n_obj=n_obj, dtype=dtype, objective_tolerance=tol)
    elif pruning == "hv_contrib":
        manager = HypervolumeArchive(capacity, n_var, n_obj, dtype, objective_tolerance=tol)
    else:
        manager = CrowdingDistanceArchive(capacity, n_var, n_obj, dtype, objective_tolerance=tol)

    archive_X, archive_F = manager.update(X, F)
    return archive_X, archive_F, manager


def update_archive(
    state: AlgorithmState,
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


__all__ = ["setup_archive", "update_archive", "resolve_external_archive"]
