# algorithm/ibea/setup.py
"""
Setup and initialization helpers for IBEA.

This module contains functions for parsing configuration, initializing populations,
and setting up archives and trackers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.engine.algorithm.components.archives import resolve_archive_size, setup_archive
from vamos.engine.algorithm.components.hooks import get_live_viz, setup_genealogy
from vamos.engine.algorithm.components.lifecycle import get_eval_strategy
from vamos.engine.algorithm.components.metrics import setup_hv_tracker
from vamos.engine.algorithm.components.termination import parse_termination
from vamos.foundation.eval.population import evaluate_population_with_constraints
from vamos.engine.algorithm.components.population import (
    evaluate_population,
    initialize_population,
    resolve_bounds,
)

from .helpers import environmental_selection
from .operators import build_variation_pipeline
from .state import IBEAState

if TYPE_CHECKING:
    from vamos.foundation.eval.backends import EvaluationBackend
    from vamos.foundation.kernel.backend import KernelBackend
    from vamos.foundation.problem.types import ProblemProtocol
from vamos.hooks.live_viz import LiveVisualization
from vamos.foundation.observer import RunContext


def initialize_ibea_run(
    cfg: dict[str, Any],
    kernel: "KernelBackend",
    problem: "ProblemProtocol",
    termination: tuple[str, Any],
    seed: int,
    eval_strategy: "EvaluationBackend | None" = None,
    live_viz: "LiveVisualization | None" = None,
) -> tuple[IBEAState, Any, Any, int, Any]:
    """Initialize all components for an IBEA run.

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
    eval_strategy : EvaluationBackend | None
        Optional evaluation backend.
    live_viz : LiveVisualization | None
        Optional live visualization callback.

    Returns
    -------
    tuple[IBEAState, Any, Any, int, Any]
        (state, live_cb, eval_strategy, max_eval, hv_tracker)
    """
    max_eval, hv_config = parse_termination(termination, cfg)

    live_cb = get_live_viz(live_viz)
    eval_strategy = get_eval_strategy(eval_strategy)

    # Setup HV tracker if configured
    hv_tracker = None
    if hv_config is not None:
        hv_tracker = setup_hv_tracker(hv_config, problem.n_obj)

    rng = np.random.default_rng(seed)
    pop_size = int(cfg["pop_size"])
    offspring_size = pop_size
    encoding = getattr(problem, "encoding", "continuous")
    n_var = problem.n_var
    n_obj = problem.n_obj
    xl, xu = resolve_bounds(problem, encoding)
    constraint_mode = cfg.get("constraint_mode", "none")

    # Initialize population
    initializer_cfg = cfg.get("initializer")
    X = initialize_population(pop_size, n_var, xl, xu, encoding, rng, problem, initializer=initializer_cfg)

    if constraint_mode and constraint_mode != "none":
        F, G = evaluate_population_with_constraints(problem, X)
    else:
        F = evaluate_population(problem, X)
        G = None
    n_eval = X.shape[0]

    # Selection pressure
    sel_method, sel_params = cfg["selection"]
    pressure = int(sel_params.get("pressure", 2))

    # Build variation pipeline
    variation = build_variation_pipeline(cfg, encoding, n_var, xl, xu, problem)

    # IBEA parameters
    indicator = cfg.get("indicator", "eps").lower()
    if indicator in ("eps", "epsilon", "additive_epsilon"):
        indicator = "epsilon"
    kappa = float(cfg.get("kappa", 1.0))

    # Compute initial fitness
    _, _, _, fitness = environmental_selection(X.copy(), F.copy(), G.copy() if G is not None else None, pop_size, indicator, kappa)

    # Setup external archive
    archive_size = resolve_archive_size(cfg) or 0
    archive_manager = None
    archive_X: np.ndarray | None = None
    archive_F: np.ndarray | None = None

    if archive_size > 0:
        archive_type = cfg.get("archive_type", "hypervolume")
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

    # Setup genealogy
    track_genealogy = bool(cfg.get("track_genealogy", False))
    genealogy_tracker, ids = setup_genealogy(pop_size, F, track_genealogy, "ibea")

    ctx = RunContext(
        problem=problem,
        algorithm=None,
        config=cfg,
        algorithm_name="ibea",
        engine_name=str(cfg.get("engine", "unknown")),
    )
    live_cb.on_start(ctx)

    # Create state
    state = IBEAState(
        X=X,
        F=F,
        G=G,
        rng=rng,
        pop_size=pop_size,
        offspring_size=offspring_size,
        constraint_mode=constraint_mode,
        n_eval=n_eval,
        # IBEA specific
        indicator=indicator,
        kappa=kappa,
        fitness=fitness,
        pressure=pressure,
        variation=variation,
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
    )

    return state, live_cb, eval_strategy, max_eval, hv_tracker


__all__ = [
    "initialize_ibea_run",
]
