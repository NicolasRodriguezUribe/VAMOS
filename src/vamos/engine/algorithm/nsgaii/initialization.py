# algorithm/nsgaii/setup.py
"""
Setup and initialization helpers for NSGA-II.

This module contains functions for parsing configuration, initializing populations,
archives, and genealogy tracking. These are extracted from the main NSGAII class
to keep the core algorithm focused on the evolutionary loop.

Operator pool building lives in operators/policies/nsgaii.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vamos.archive import ExternalArchiveConfig
from vamos.engine.algorithm.components.archive import (
    CrowdingDistanceArchive,
    HypervolumeArchive,
    UnboundedArchive,
)
from vamos.engine.algorithm.components.population import initialize_population, resolve_bounds
from vamos.foundation.encoding import normalize_encoding
from vamos.foundation.eval.backends import EvaluationBackend
from vamos.foundation.kernel.backend import KernelBackend
from vamos.foundation.problem.types import ProblemProtocol
from vamos.hooks.genealogy import DefaultGenealogyTracker, GenealogyTracker

# Constants
DEFAULT_TOURNAMENT_PRESSURE = 2


def parse_termination(termination: tuple[str, Any]) -> tuple[int, dict[str, Any] | None]:
    """Parse termination criterion and return (max_eval, hv_config).

    Parameters
    ----------
    termination : tuple[str, Any]
        Termination criterion as (type, value). Supported types:
        - "max_evaluations": value is the max number of evaluations
        - "hv": value is a dict with hypervolume config

    Returns
    -------
    tuple[int, dict[str, Any] | None]
        (max_evaluations, hv_config or None)

    Raises
    ------
    ValueError
        If termination type is unsupported or HV config is invalid.
    """
    term_type, term_val = termination
    hv_config = None
    if term_type == "max_evaluations":
        max_eval = int(term_val)
    elif term_type == "hv":
        hv_config = dict(term_val)
        max_eval = int(hv_config.get("max_evaluations", 0))
        if max_eval <= 0:
            raise ValueError("HV-based termination requires a positive max_evaluations value.")
    else:
        raise ValueError("Unsupported termination criterion for NSGA-II.")
    return max_eval, hv_config


def setup_population(
    problem: ProblemProtocol,
    eval_strategy: EvaluationBackend,
    rng: np.random.Generator,
    pop_size: int,
    constraint_mode: str,
    initializer_cfg: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
    """Initialize and evaluate the starting population.

    Parameters
    ----------
    problem : ProblemProtocol
        The optimization problem.
    eval_strategy : EvaluationBackend
        Backend for evaluating solutions.
    rng : np.random.Generator
        Random number generator.
    pop_size : int
        Population size.
    constraint_mode : str
        Constraint handling mode ('none', 'feasibility', etc.).
    initializer_cfg : dict[str, Any] | None
        Optional initializer configuration.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray | None, int]
        (X, F, G, n_eval) - decision variables, objectives, constraints, evaluation count.
    """
    encoding = normalize_encoding(getattr(problem, "encoding", "real"))
    n_var = problem.n_var
    xl, xu = resolve_bounds(problem, encoding)
    X = initialize_population(pop_size, n_var, xl, xu, encoding, rng, problem, initializer=initializer_cfg)
    eval_result = eval_strategy.evaluate(X, problem)
    F = eval_result.F
    G = eval_result.G if constraint_mode != "none" else None
    assert X.shape[0] == F.shape[0], "Population and objectives must align"
    return X, F, G, X.shape[0]


def setup_archive(
    kernel: KernelBackend,
    X: np.ndarray,
    F: np.ndarray,
    n_var: int,
    n_obj: int,
    dtype: np.dtype,
    ext_cfg: ExternalArchiveConfig | None,
) -> tuple[np.ndarray | None, np.ndarray | None, CrowdingDistanceArchive | HypervolumeArchive | UnboundedArchive | None]:
    """Initialize archive if configured.

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
    tuple[np.ndarray | None, np.ndarray | None, archive_manager | None]
        (archive_X, archive_F, archive_manager)
    """
    if ext_cfg is None:
        return None, None, None

    capacity = ext_cfg.capacity
    pruning = ext_cfg.pruning

    manager: CrowdingDistanceArchive | HypervolumeArchive | UnboundedArchive
    if capacity is None:
        # Unbounded archive
        manager = UnboundedArchive(n_var=n_var, n_obj=n_obj, dtype=dtype)
    elif pruning == "hv_contrib":
        manager = HypervolumeArchive(capacity, n_var, n_obj, dtype)
    else:
        manager = CrowdingDistanceArchive(capacity, n_var, n_obj, dtype)

    archive_X, archive_F = manager.update(X, F)
    return archive_X, archive_F, manager


def setup_genealogy(
    pop_size: int,
    F: np.ndarray,
    track_genealogy: bool,
) -> tuple[GenealogyTracker | None, np.ndarray | None]:
    """Initialize genealogy tracking if enabled.

    Parameters
    ----------
    pop_size : int
        Population size.
    F : np.ndarray
        Initial fitness values.
    track_genealogy : bool
        Whether to enable genealogy tracking.

    Returns
    -------
    tuple[GenealogyTracker | None, np.ndarray | None]
        (genealogy_tracker, individual_ids)
    """
    if not track_genealogy:
        return None, None

    genealogy_tracker = DefaultGenealogyTracker()
    ids = np.arange(pop_size, dtype=int)
    for i in range(pop_size):
        genealogy_tracker.new_individual(
            generation=0,
            parents=[],
            operator_name=None,
            algorithm_name="nsgaii",
            fitness=F[i] if F is not None and i < F.shape[0] else None,
        )
    return genealogy_tracker, ids


def setup_selection(
    sel_method: str,
    sel_params: dict[str, Any],
) -> tuple[str, int]:
    """Parse selection config and return (method, pressure).

    Parameters
    ----------
    sel_method : str
        Selection method name ('tournament' or 'random').
    sel_params : dict[str, Any]
        Selection parameters.

    Returns
    -------
    tuple[str, int]
        (method, tournament_pressure)

    Raises
    ------
    ValueError
        If selection method is not supported.
    """
    allowed = ("tournament", "random", "boltzmann", "ranking", "sus")
    if sel_method not in allowed:
        raise ValueError(f"Unsupported selection method '{sel_method}'. Must be one of {allowed}.")
    pressure = int(sel_params.get("pressure", DEFAULT_TOURNAMENT_PRESSURE)) if sel_method == "tournament" else DEFAULT_TOURNAMENT_PRESSURE
    return sel_method, pressure


def setup_result_archive(
    ext_cfg: ExternalArchiveConfig | None,
    n_var: int,
    n_obj: int,
    dtype: np.dtype,
) -> HypervolumeArchive | CrowdingDistanceArchive | None:
    """Create result archive if configured.

    Parameters
    ----------
    ext_cfg : ExternalArchiveConfig | None
        External archive configuration. The ``pruning`` field determines
        the archive class (``"hv_contrib"`` -> HypervolumeArchive,
        otherwise CrowdingDistanceArchive). ``None`` or unbounded
        (``capacity is None``) disables the result archive.
    n_var : int
        Number of decision variables.
    n_obj : int
        Number of objectives.
    dtype : np.dtype
        Data type for arrays.

    Returns
    -------
    HypervolumeArchive | CrowdingDistanceArchive | None
        The result archive, or None if not configured.
    """
    if ext_cfg is None or ext_cfg.capacity is None:
        return None

    if ext_cfg.pruning == "hv_contrib":
        return HypervolumeArchive(ext_cfg.capacity, n_var, n_obj, dtype)
    return CrowdingDistanceArchive(ext_cfg.capacity, n_var, n_obj, dtype)


def resolve_external_archive(cfg: dict[str, Any]) -> ExternalArchiveConfig | None:
    """Extract :class:`ExternalArchiveConfig` from a serialised config dict.

    Returns ``None`` when no external archive is configured.

    Parameters
    ----------
    cfg : dict[str, Any]
        Algorithm configuration.

    Returns
    -------
    ExternalArchiveConfig | None
        External archive configuration, or ``None``.
    """
    raw = cfg.get("external_archive")
    if raw is None:
        return None
    if isinstance(raw, ExternalArchiveConfig):
        return raw
    return ExternalArchiveConfig(**raw)


__all__ = [
    "parse_termination",
    "setup_population",
    "setup_archive",
    "setup_genealogy",
    "setup_selection",
    "setup_result_archive",
    "resolve_external_archive",
    "DEFAULT_TOURNAMENT_PRESSURE",
]
