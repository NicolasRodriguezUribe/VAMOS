"""
Base infrastructure for multi-objective optimization algorithms.

This module provides shared building blocks that all algorithms can use:
- AlgorithmState: Base state container with common fields
- Shared setup functions for population, archives, termination, visualization
- Ask/tell interface support

By using this shared infrastructure, all algorithms gain consistent features:
- HV-based termination
- Live visualization callbacks
- External archives
- Genealogy tracking (optional)
- Consistent result format
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.engine.algorithm.components.archive import CrowdingDistanceArchive, HypervolumeArchive
from vamos.engine.algorithm.components.population import (
    initialize_population,
    resolve_bounds,
)
from vamos.engine.algorithm.components.termination import HVTracker
from vamos.foundation.eval.backends import EvaluationBackend, SerialEvalBackend

if TYPE_CHECKING:
    from vamos.foundation.kernel.backend import KernelBackend
    from vamos.foundation.problem.types import ProblemProtocol
    from vamos.hooks.genealogy import GenealogyTracker
    from vamos.hooks.live_viz import LiveVisualization

_logger = logging.getLogger(__name__)


# =============================================================================
# Base State Container
# =============================================================================


@dataclass
class AlgorithmState:
    """
    Base state container for evolutionary algorithms.

    This provides common fields that all algorithms need. Algorithm-specific
    subclasses can add additional fields as needed.

    Attributes
    ----------
    X : np.ndarray
        Decision variables, shape (pop_size, n_var).
    F : np.ndarray
        Objective values, shape (pop_size, n_obj).
    G : np.ndarray | None
        Constraint values, shape (pop_size, n_constr) or None.
    rng : np.random.Generator
        Random number generator.
    pop_size : int
        Population size.
    offspring_size : int
        Number of offspring per generation.
    constraint_mode : str
        Constraint handling mode ('none', 'feasibility', 'penalty').
    generation : int
        Current generation number.
    n_eval : int
        Total number of evaluations so far.
    """

    # Core population data
    X: np.ndarray
    F: np.ndarray
    G: np.ndarray | None
    rng: np.random.Generator

    # Sizes
    pop_size: int = 100
    offspring_size: int = 100

    # Constraints
    constraint_mode: str = "none"

    # Generation tracking
    generation: int = 0
    n_eval: int = 0

    # Archive (optional)
    archive_size: int | None = None
    archive_X: np.ndarray | None = None
    archive_F: np.ndarray | None = None
    archive_manager: CrowdingDistanceArchive | HypervolumeArchive | None = None
    result_mode: str = "non_dominated"

    # Termination
    hv_tracker: HVTracker | None = None

    # Pending offspring for ask/tell
    pending_offspring: np.ndarray | None = None
    pending_offspring_ids: np.ndarray | None = None

    # Genealogy (optional)
    track_genealogy: bool = False
    genealogy_tracker: "GenealogyTracker | None" = None
    ids: np.ndarray | None = None

    def hv_points(self) -> np.ndarray:
        """Get points for hypervolume computation (archive if available, else population)."""
        if self.archive_F is not None and self.archive_F.size > 0:
            return self.archive_F
        return self.F


# =============================================================================
# Termination Parsing
# =============================================================================


def parse_termination(
    termination: tuple[str, Any],
    algorithm_name: str = "algorithm",
) -> tuple[int, dict[str, Any] | None]:
    """
    Parse termination criterion and return (max_eval, hv_config).

    Parameters
    ----------
    termination : tuple[str, Any]
        Termination criterion as (type, value). Supported types:
        - "n_eval": value is the max number of evaluations
        - "hv": value is a dict with hypervolume config
    algorithm_name : str
        Algorithm name for error messages.

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
            raise ValueError(
                f"HV-based termination for {algorithm_name} requires a positive max_evaluations value."
            )
    else:
        raise ValueError(f"Unsupported termination criterion '{term_type}' for {algorithm_name}.")

    return max_eval, hv_config


# =============================================================================
# Population Setup
# =============================================================================


def setup_initial_population(
    problem: "ProblemProtocol",
    eval_backend: EvaluationBackend,
    rng: np.random.Generator,
    pop_size: int,
    constraint_mode: str,
    initializer_cfg: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
    """
    Initialize and evaluate the starting population.

    Parameters
    ----------
    problem : ProblemProtocol
        The optimization problem.
    eval_backend : EvaluationBackend
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
    encoding = getattr(problem, "encoding", "continuous")
    n_var = problem.n_var
    xl, xu = resolve_bounds(problem, encoding)

    X = initialize_population(
        pop_size, n_var, xl, xu, encoding, rng, problem, initializer=initializer_cfg
    )
    eval_result = eval_backend.evaluate(X, problem)
    F = eval_result.F
    G = eval_result.G if constraint_mode != "none" else None

    return X, F, G, X.shape[0]


# =============================================================================
# Archive Setup
# =============================================================================


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


# =============================================================================
# HV Tracker Setup
# =============================================================================


def setup_hv_tracker(
    hv_config: dict[str, Any] | None,
    kernel: "KernelBackend",
) -> HVTracker:
    """
    Create HV tracker from config.

    Parameters
    ----------
    hv_config : dict[str, Any] | None
        HV termination configuration.
    kernel : KernelBackend
        Kernel backend.

    Returns
    -------
    HVTracker
        Configured tracker (may be disabled if config is None).
    """
    return HVTracker(hv_config, kernel)


# =============================================================================
# Live Visualization
# =============================================================================


def get_live_viz(
    live_viz: "LiveVisualization | None",
) -> "LiveVisualization":
    """
    Get live visualization callback, defaulting to no-op.

    Parameters
    ----------
    live_viz : LiveVisualization | None
        User-provided callback or None.

    Returns
    -------
    LiveVisualization
        The callback or a no-op implementation.
    """
    from vamos.hooks.live_viz import NoOpLiveVisualization

    return live_viz or NoOpLiveVisualization()


def notify_generation(
    live_cb: "LiveVisualization",
    kernel: "KernelBackend",
    generation: int,
    F: np.ndarray,
    stats: dict[str, Any] | None = None,
) -> bool:
    """
    Notify live visualization of generation progress.

    Parameters
    ----------
    live_cb : LiveVisualization
        Live visualization callback.
    kernel : KernelBackend
        Kernel for computing non-dominated front.
    generation : int
        Current generation number.
    F : np.ndarray
        Current population objectives.
    stats : dict[str, Any] | None
        Optional metrics payload (e.g., evaluation counts).

    Returns
    -------
    bool
        True if the callback requests stopping.
    """
    try:
        ranks, _ = kernel.nsga2_ranking(F)
        nd_mask = ranks == ranks.min(initial=0)
        live_cb.on_generation(generation, F=F[nd_mask], stats=stats)
    except (ValueError, IndexError) as exc:
        _logger.debug("Failed to compute non-dominated front for viz: %s", exc)
        live_cb.on_generation(generation, F=F, stats=stats)
    return live_should_stop(live_cb)


def live_should_stop(live_cb: "LiveVisualization") -> bool:
    should_stop = getattr(live_cb, "should_stop", None)
    if not callable(should_stop):
        return False
    try:
        return bool(should_stop())
    except Exception:
        return False


# =============================================================================
# Result Building
# =============================================================================


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

    # Add archive contents
    if state.archive_manager is not None:
        arch_X, arch_F = state.archive_manager.contents()
        result["archive"] = {"X": arch_X, "F": arch_F}
    elif state.archive_X is not None and state.archive_F is not None:
        result["archive"] = {"X": state.archive_X, "F": state.archive_F}

    # Merge extra fields
    if extra:
        result.update(extra)

    return result


# =============================================================================
# Evaluation Backend
# =============================================================================


def get_eval_backend(
    eval_backend: EvaluationBackend | None,
) -> EvaluationBackend:
    """
    Get evaluation backend, defaulting to serial.

    Parameters
    ----------
    eval_backend : EvaluationBackend | None
        User-provided backend or None.

    Returns
    -------
    EvaluationBackend
        The backend or a serial implementation.
    """
    return eval_backend or SerialEvalBackend()


# =============================================================================
# Genealogy Setup
# =============================================================================


def setup_genealogy(
    pop_size: int,
    F: np.ndarray,
    track_genealogy: bool,
    algorithm_name: str = "algorithm",
) -> tuple["GenealogyTracker | None", np.ndarray | None]:
    """
    Initialize genealogy tracking if enabled.

    Parameters
    ----------
    pop_size : int
        Population size.
    F : np.ndarray
        Initial fitness values.
    track_genealogy : bool
        Whether to enable genealogy tracking.
    algorithm_name : str
        Algorithm name for tracking records.

    Returns
    -------
    tuple[GenealogyTracker | None, np.ndarray | None]
        (genealogy_tracker, individual_ids)
    """
    if not track_genealogy:
        return None, None

    from vamos.hooks.genealogy import DefaultGenealogyTracker

    genealogy_tracker = DefaultGenealogyTracker()
    ids = np.arange(pop_size, dtype=int)
    for i in range(pop_size):
        genealogy_tracker.new_individual(
            generation=0,
            parents=[],
            operator_name=None,
            algorithm_name=algorithm_name,
            fitness=F[i] if F is not None and i < F.shape[0] else None,
        )
    return genealogy_tracker, ids


def track_offspring_genealogy(
    state: AlgorithmState,
    parent_idx: np.ndarray,
    offspring_count: int,
    operator_name: str = "variation",
    algorithm_name: str = "algorithm",
) -> None:
    """
    Record genealogy for new offspring.

    Parameters
    ----------
    state : AlgorithmState
        Algorithm state with genealogy fields.
    parent_idx : np.ndarray
        Indices of parents used to create offspring.
    offspring_count : int
        Number of offspring created.
    operator_name : str
        Name of the variation operator used.
    algorithm_name : str
        Algorithm name for tracking.
    """
    if not state.track_genealogy or state.genealogy_tracker is None:
        return

    parents_per_offspring = max(1, len(parent_idx) // offspring_count)
    pending_ids = []

    for i in range(offspring_count):
        start = i * parents_per_offspring
        end = min(start + parents_per_offspring, len(parent_idx))
        parent_ids = []
        if state.ids is not None:
            parent_ids = [int(state.ids[p]) for p in parent_idx[start:end] if p < len(state.ids)]

        new_id = state.genealogy_tracker.new_individual(
            generation=state.generation + 1,
            parents=parent_ids,
            operator_name=operator_name,
            algorithm_name=algorithm_name,
            fitness=None,
        )
        pending_ids.append(new_id)

    state.pending_offspring_ids = np.array(pending_ids, dtype=int)


def finalize_genealogy(
    result: dict,
    state: AlgorithmState,
    kernel: "KernelBackend",
) -> None:
    """
    Finalize genealogy tracking and add to result.

    Parameters
    ----------
    result : dict
        Result dictionary to update.
    state : AlgorithmState
        Algorithm state with genealogy tracker.
    kernel : KernelBackend
        Kernel backend for ranking computation.
    """
    if not state.track_genealogy or state.genealogy_tracker is None:
        return

    try:
        # Import here to avoid circular imports
        from vamos.engine.algorithm.nsgaii.helpers import (
            generation_contributions,
            operator_success_stats,
        )

        # Identify final front for genealogy stats
        ranks, _ = kernel.nsga2_ranking(state.F)
        nd_mask = ranks == ranks.min(initial=0)
        final_front_ids = state.ids[nd_mask] if state.ids is not None else []
        state.genealogy_tracker.mark_final_front(list(final_front_ids))

        result["genealogy"] = {
            "operator_stats": operator_success_stats(
                state.genealogy_tracker, list(final_front_ids)
            ),
            "generation_contributions": generation_contributions(
                state.genealogy_tracker, list(final_front_ids)
            ),
        }
    except (ValueError, IndexError, AttributeError) as exc:
        import logging

        logging.getLogger(__name__).warning("Failed to compute genealogy stats: %s", exc)


def match_ids(
    new_X: np.ndarray,
    combined_X: np.ndarray,
    combined_ids: np.ndarray,
) -> np.ndarray:
    """
    Match IDs after survival selection by finding matching rows.

    Parameters
    ----------
    new_X : np.ndarray
        New population after survival selection.
    combined_X : np.ndarray
        Combined parent+offspring population before selection.
    combined_ids : np.ndarray
        IDs corresponding to combined_X.

    Returns
    -------
    np.ndarray
        IDs for the new population.
    """
    new_ids = np.zeros(new_X.shape[0], dtype=int)
    for i, row in enumerate(new_X):
        for j, comb_row in enumerate(combined_X):
            if np.allclose(row, comb_row, rtol=1e-12, atol=1e-12):
                new_ids[i] = combined_ids[j]
                break
    return new_ids


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # State
    "AlgorithmState",
    # Setup functions
    "parse_termination",
    "setup_initial_population",
    "setup_archive",
    "update_archive",
    "resolve_archive_size",
    "setup_hv_tracker",
    "get_live_viz",
    "notify_generation",
    "live_should_stop",
    "get_eval_backend",
    # Genealogy
    "setup_genealogy",
    "track_offspring_genealogy",
    "finalize_genealogy",
    "match_ids",
    # Result
    "build_result",
]
