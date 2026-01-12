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

from vamos.engine.algorithm.components.archive import CrowdingDistanceArchive, HypervolumeArchive
from vamos.engine.algorithm.components.population import initialize_population, resolve_bounds
from vamos.foundation.encoding import normalize_encoding
from vamos.hooks.genealogy import DefaultGenealogyTracker, GenealogyTracker
from vamos.foundation.eval.backends import EvaluationBackend
from vamos.foundation.kernel.backend import KernelBackend
from vamos.foundation.problem.types import ProblemProtocol

# Constants
DEFAULT_TOURNAMENT_PRESSURE = 2


def parse_termination(termination: tuple[str, Any]) -> tuple[int, dict[str, Any] | None]:
    """Parse termination criterion and return (max_eval, hv_config).

    Parameters
    ----------
    termination : tuple[str, Any]
        Termination criterion as (type, value). Supported types:
        - "n_eval": value is the max number of evaluations
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
    if term_type == "n_eval":
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
    archive_size: int | None,
) -> tuple[np.ndarray | None, np.ndarray | None, CrowdingDistanceArchive | None, bool]:
    """Initialize archive if configured.

    Parameters
    ----------
    kernel : KernelBackend
        The kernel backend (may have update_archive method).
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

    Returns
    -------
    tuple[np.ndarray | None, np.ndarray | None, CrowdingDistanceArchive | None, bool]
        (archive_X, archive_F, archive_manager, archive_via_kernel)
    """
    if not archive_size:
        return None, None, None, False

    if hasattr(kernel, "update_archive"):
        archive_X, archive_F = kernel.update_archive(None, None, X, F, archive_size)
        return archive_X, archive_F, None, True
    else:
        archive_manager = CrowdingDistanceArchive(archive_size, n_var, n_obj, dtype)
        archive_X, archive_F = archive_manager.update(X, F)
        return archive_X, archive_F, archive_manager, False


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
    if sel_method not in ("tournament", "random"):
        raise ValueError(f"Unsupported selection method '{sel_method}'.")
    pressure = int(sel_params.get("pressure", DEFAULT_TOURNAMENT_PRESSURE)) if sel_method == "tournament" else DEFAULT_TOURNAMENT_PRESSURE
    return sel_method, pressure


def setup_result_archive(
    result_mode: str,
    archive_type: str,
    archive_size: int | None,
    n_var: int,
    n_obj: int,
    dtype: np.dtype,
) -> HypervolumeArchive | CrowdingDistanceArchive | None:
    """Create result archive if configured for external_archive mode.

    Parameters
    ----------
    result_mode : str
        Result mode ('population' or 'external_archive').
    archive_type : str
        Archive type ('hypervolume' or 'crowding').
    archive_size : int | None
        Archive capacity.
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
    if result_mode != "external_archive" or not archive_size:
        return None

    if archive_type == "crowding":
        return CrowdingDistanceArchive(archive_size, n_var, n_obj, dtype)
    return HypervolumeArchive(archive_size, n_var, n_obj, dtype)


def resolve_archive_size(cfg: dict[str, Any]) -> int | None:
    """Resolve archive size from configuration.

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


__all__ = [
    "parse_termination",
    "setup_population",
    "setup_archive",
    "setup_genealogy",
    "setup_selection",
    "setup_result_archive",
    "resolve_archive_size",
    "DEFAULT_TOURNAMENT_PRESSURE",
]
