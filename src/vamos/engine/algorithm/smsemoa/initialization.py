"""SMS-EMOA initialization and setup routines.

This module handles algorithm setup including:
- Parsing termination criteria
- Initial population generation
- Reference point setup
- Archive configuration
- Genealogy tracking setup
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.engine.algorithm.components.archives import resolve_archive_size, setup_archive
from vamos.engine.algorithm.components.hooks import get_live_viz, setup_genealogy
from vamos.engine.algorithm.components.lifecycle import get_eval_strategy
from vamos.engine.algorithm.components.metrics import setup_hv_tracker
from vamos.engine.algorithm.components.termination import parse_termination
from vamos.engine.algorithm.components.utils import resolve_bounds_array
from .helpers import (
    evaluate_population_with_constraints,
    initialize_reference_point,
)
from .operators import build_variation_operators
from .state import SMSEMOAState
from vamos.operators.binary import random_binary_population
from vamos.operators.integer import random_integer_population

if TYPE_CHECKING:
    from vamos.foundation.eval.backends import EvaluationBackend
    from vamos.foundation.kernel.protocols import KernelBackend
    from vamos.foundation.problem.types import ProblemProtocol
    from vamos.hooks.live_viz import LiveVisualization


__all__ = [
    "initialize_smsemoa_run",
    "initialize_population",
]


def initialize_smsemoa_run(
    config: dict,
    kernel: "KernelBackend",
    problem: "ProblemProtocol",
    termination: tuple[str, Any],
    seed: int,
    eval_strategy: "EvaluationBackend | None" = None,
    live_viz: "LiveVisualization | None" = None,
) -> tuple["SMSEMOAState", Any, Any, int, Any]:
    """Initialize SMS-EMOA run and create state.

    Parameters
    ----------
    config : dict
        Algorithm configuration.
    kernel : KernelBackend
        Backend for vectorized operations.
    problem : ProblemProtocol
        Problem to optimize.
    termination : tuple
        Termination criterion, e.g., ("n_eval", 10000).
    seed : int
        Random seed for reproducibility.
    eval_strategy : EvaluationBackend, optional
        Evaluation backend for parallel evaluation.
    live_viz : LiveVisualization, optional
        Live visualization callback.

    Returns
    -------
    tuple
        (state, live_cb, eval_strategy, max_eval, hv_tracker)
    """
    max_eval, hv_config = parse_termination(termination, config)

    live_cb = get_live_viz(live_viz)
    eval_strategy = get_eval_strategy(eval_strategy)

    # Setup HV tracker if configured
    hv_tracker = None
    if hv_config is not None:
        hv_tracker = setup_hv_tracker(hv_config, problem.n_obj)

    rng = np.random.default_rng(seed)
    pop_size = config["pop_size"]
    encoding = getattr(problem, "encoding", "continuous")
    xl, xu = resolve_bounds_array(problem, encoding)
    n_var = problem.n_var
    n_obj = problem.n_obj
    constraint_mode = config.get("constraint_mode", "penalty")

    # Build variation operators
    crossover_fn, mutation_fn = build_variation_operators(config, encoding, n_var, xl, xu, rng)

    # Selection pressure
    sel_method, sel_params = config["selection"]
    pressure = sel_params.get("pressure", 2) if sel_method == "tournament" else 2

    # Reference point config
    ref_cfg = config.get("reference_point", {}) or {}

    # Initialize population
    X, F, G = initialize_population(encoding, pop_size, n_var, xl, xu, rng, problem, constraint_mode)
    n_eval = pop_size

    # Setup genealogy tracking
    track_genealogy = bool(config.get("track_genealogy", False))
    genealogy_tracker, ids = setup_genealogy(pop_size, F, track_genealogy, "smsemoa")

    # Initialize reference point
    ref_point, ref_offset, ref_adaptive = initialize_reference_point(F, ref_cfg)

    # Setup external archive
    archive_size = resolve_archive_size(config) or 0
    archive_manager = None
    archive_X: np.ndarray | None = None
    archive_F: np.ndarray | None = None

    if archive_size > 0:
        archive_type = config.get("archive_type", "hypervolume")
        archive_manager, archive_X, archive_F = setup_archive(
            kernel=kernel,
            X=X,
            F=F,
            n_var=n_var,
            n_obj=n_obj,
            dtype=X.dtype,
            archive_size=archive_size,
            archive_type=archive_type,
        )

    # Create state
    state = SMSEMOAState(
        X=X,
        F=F,
        G=G,
        rng=rng,
        pop_size=pop_size,
        offspring_size=1,  # SMS-EMOA typically generates 1 offspring per iteration
        constraint_mode=constraint_mode,
        n_eval=n_eval,
        # SMSEMOA-specific
        ref_point=ref_point,
        ref_offset=ref_offset,
        ref_adaptive=ref_adaptive,
        pressure=pressure,
        crossover_fn=crossover_fn,
        mutation_fn=mutation_fn,
        # Archive
        archive_size=archive_size,
        archive_X=archive_X,
        archive_F=archive_F,
        archive_manager=archive_manager,
        # Termination
        hv_tracker=hv_tracker,
        # Genealogy
        track_genealogy=track_genealogy,
        genealogy_tracker=genealogy_tracker,
        ids=ids,
        result_mode=config.get("result_mode", "non_dominated"),
    )

    return state, live_cb, eval_strategy, max_eval, hv_tracker


def initialize_population(
    encoding: str,
    pop_size: int,
    n_var: int,
    xl: np.ndarray,
    xu: np.ndarray,
    rng: np.random.Generator,
    problem: "ProblemProtocol",
    constraint_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Initialize population based on encoding.

    Parameters
    ----------
    encoding : str
        Variable encoding type.
    pop_size : int
        Population size.
    n_var : int
        Number of variables.
    xl : np.ndarray
        Lower bounds.
    xu : np.ndarray
        Upper bounds.
    rng : np.random.Generator
        Random number generator.
    problem : ProblemProtocol
        Problem for evaluation.
    constraint_mode : str
        Constraint handling mode.

    Returns
    -------
    tuple
        (X, F, G) initial population, objectives, and constraints.
    """
    if encoding == "binary":
        X = random_binary_population(pop_size, n_var, rng)
    elif encoding == "integer":
        X = random_integer_population(pop_size, n_var, xl.astype(int), xu.astype(int), rng)
    else:
        X = rng.uniform(xl, xu, size=(pop_size, n_var))

    F, G = evaluate_population_with_constraints(problem, X)
    if constraint_mode == "none":
        G = None
    return X, F, G
