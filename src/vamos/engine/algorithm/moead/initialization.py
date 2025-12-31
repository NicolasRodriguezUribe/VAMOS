# algorithm/moead/setup.py
"""
Setup and initialization helpers for MOEA/D.

This module contains functions for parsing configuration, initializing populations,
weight vectors, neighborhoods, and other setup tasks.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.engine.algorithm.components.base import (
    get_eval_backend,
    get_live_viz,
    parse_termination,
    resolve_archive_size,
    setup_archive,
    setup_genealogy,
    setup_hv_tracker,
)
from vamos.foundation.eval.population import evaluate_population_with_constraints
from vamos.engine.algorithm.components.utils import resolve_bounds_array
from vamos.engine.algorithm.components.weight_vectors import load_or_generate_weight_vectors
from vamos.operators.binary import random_binary_population
from vamos.operators.integer import random_integer_population

from .helpers import build_aggregator, compute_neighbors
from .operators import build_variation_operators
from .state import MOEADState

if TYPE_CHECKING:
    from vamos.engine.algorithm.components.termination import HVTracker
    from vamos.foundation.eval.backends import EvaluationBackend
    from vamos.foundation.kernel.backend import KernelBackend
    from vamos.foundation.problem.types import ProblemProtocol
from vamos.hooks.live_viz import LiveVisualization


def initialize_moead_run(
    cfg: dict[str, Any],
    kernel: "KernelBackend",
    problem: "ProblemProtocol",
    termination: tuple[str, Any],
    seed: int,
    eval_backend: "EvaluationBackend | None" = None,
    live_viz: "LiveVisualization | None" = None,
) -> tuple[MOEADState, Any, Any, int, "HVTracker"]:
    """Initialize all components for a MOEA/D run.

    Parameters
    ----------
    cfg : dict[str, Any]
        Algorithm configuration.
    kernel : KernelBackend
        Backend for vectorized operations.
    problem : ProblemProtocol
        The optimization problem.
    termination : tuple[str, Any]
        Termination criterion.
    seed : int
        Random seed.
    eval_backend : EvaluationBackend | None
        Optional evaluation backend.
    live_viz : LiveVisualization | None
        Optional live visualization callback.

    Returns
    -------
    tuple[MOEADState, Any, Any, int, HVTracker]
        (state, live_cb, eval_backend, max_eval, hv_tracker)
    """
    max_eval, hv_config = parse_termination(termination, "MOEA/D")

    eval_backend = get_eval_backend(eval_backend)
    live_cb = get_live_viz(live_viz)
    rng = np.random.default_rng(seed)

    pop_size = int(cfg["pop_size"])
    if pop_size < 2:
        raise ValueError("MOEA/D requires pop_size >= 2.")

    constraint_mode = cfg.get("constraint_mode", "feasibility")
    encoding = getattr(problem, "encoding", "continuous")
    xl, xu = resolve_bounds_array(problem, encoding)
    n_var = problem.n_var
    n_obj = problem.n_obj

    # Initialize population
    X, F, G = initialize_population(
        encoding, pop_size, n_var, xl, xu, rng, problem, constraint_mode
    )
    n_eval = pop_size

    # Setup weight vectors and neighborhoods
    weight_cfg = cfg.get("weight_vectors", {}) or {}
    weights = load_or_generate_weight_vectors(
        pop_size, n_obj,
        path=weight_cfg.get("path"),
        divisions=weight_cfg.get("divisions"),
    )

    neighbor_size = cfg.get("neighbor_size", min(20, pop_size))
    neighbor_size = max(2, min(neighbor_size, pop_size))
    neighbors = compute_neighbors(weights, neighbor_size)

    # Setup aggregation
    aggregation = cfg.get("aggregation", ("tchebycheff", {}))
    agg_method, agg_params = aggregation
    aggregator = build_aggregator(agg_method, agg_params)

    # Build variation operators
    crossover_fn, mutation_fn = build_variation_operators(
        cfg, encoding, n_var, xl, xu, rng
    )

    # Setup archive
    archive_size = resolve_archive_size(cfg)
    archive_type = cfg.get("archive_type", "crowding")
    archive_X, archive_F, archive_manager = setup_archive(
        kernel, X, F, n_var, n_obj, X.dtype, archive_size, archive_type
    )

    # Setup HV tracker
    hv_tracker = setup_hv_tracker(hv_config, kernel)

    # Setup genealogy
    track_genealogy = bool(cfg.get("track_genealogy", False))
    genealogy_tracker, ids = setup_genealogy(pop_size, F, track_genealogy, "moead")

    live_cb.on_start(problem=problem, algorithm=None, config=cfg)

    # Create state
    state = MOEADState(
        X=X, F=F, G=G, rng=rng,
        pop_size=pop_size,
        offspring_size=pop_size,
        constraint_mode=constraint_mode,
        n_eval=n_eval,
        # MOEA/D-specific
        weights=weights,
        neighbors=neighbors,
        ideal=F.min(axis=0),
        aggregator=aggregator,
        neighbor_size=neighbor_size,
        delta=float(cfg.get("delta", 0.9)),
        replace_limit=max(1, int(cfg.get("replace_limit", 2))),
        crossover_fn=crossover_fn,
        mutation_fn=mutation_fn,
        xl=xl,
        xu=xu,
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
        result_mode=cfg.get("result_mode", "non_dominated"),
    )

    return state, live_cb, eval_backend, max_eval, hv_tracker


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
        Problem encoding: "binary", "integer", or "continuous"/"real".
    pop_size : int
        Population size.
    n_var : int
        Number of decision variables.
    xl : np.ndarray
        Lower bounds.
    xu : np.ndarray
        Upper bounds.
    rng : np.random.Generator
        Random number generator.
    problem : ProblemProtocol
        The optimization problem.
    constraint_mode : str
        Constraint handling mode.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray | None]
        (X, F, G) population arrays.
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


__all__ = [
    "initialize_moead_run",
    "initialize_population",
]
