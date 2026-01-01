"""SMPSO initialization and setup routines.

This module handles algorithm setup including:
- Population initialization
- Velocity initialization
- Leader archive setup
- Genealogy tracking setup
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.engine.algorithm.components.archive import CrowdingDistanceArchive
from vamos.engine.algorithm.components.hooks import get_live_viz, setup_genealogy
from vamos.engine.algorithm.components.lifecycle import get_eval_backend
from vamos.engine.algorithm.components.metrics import setup_hv_tracker
from vamos.engine.algorithm.components.termination import parse_termination
from vamos.engine.algorithm.components.population import (
    initialize_population,
    resolve_bounds,
)
from .operators import (
    build_mutation_operator,
    build_repair_operator,
)
from .state import SMPSOState

if TYPE_CHECKING:
    from vamos.foundation.eval.backends import EvaluationBackend
    from vamos.foundation.kernel.protocols import KernelBackend
    from vamos.foundation.problem.types import ProblemProtocol
    from vamos.hooks.live_viz import LiveVisualization


__all__ = [
    "initialize_smpso_run",
]


def initialize_smpso_run(
    config: dict,
    kernel: "KernelBackend",
    problem: "ProblemProtocol",
    termination: tuple[str, Any],
    seed: int,
    eval_backend: "EvaluationBackend | None" = None,
    live_viz: "LiveVisualization | None" = None,
) -> tuple["SMPSOState", Any, Any, int, Any]:
    """Initialize SMPSO run and create state.

    Parameters
    ----------
    config : dict
        Algorithm configuration.
    kernel : KernelBackend
        Backend for vectorized operations.
    problem : ProblemProtocol
        Problem to optimize.
    termination : tuple
        Termination criterion.
    seed : int
        Random seed.
    eval_backend : EvaluationBackend, optional
        Evaluation backend.
    live_viz : LiveVisualization, optional
        Visualization callback.

    Returns
    -------
    tuple
        (state, live_cb, eval_backend, max_eval, hv_tracker)
    """
    max_eval, hv_config = parse_termination(termination, "SMPSO")
    eval_backend = get_eval_backend(eval_backend)
    live_cb = get_live_viz(live_viz)
    rng = np.random.default_rng(seed)

    pop_size = int(config.get("pop_size", 100))
    archive_size = int(config.get("archive_size", pop_size))
    inertia = float(config.get("inertia", 0.5))
    c1 = float(config.get("c1", 1.5))
    c2 = float(config.get("c2", 1.5))
    vmax_fraction = float(config.get("vmax_fraction", 0.5))

    encoding = getattr(problem, "encoding", "continuous")
    if encoding not in {"continuous", "real"}:
        raise ValueError("SMPSO currently supports continuous/real encoding only.")

    n_var = int(problem.n_var)
    n_obj = int(problem.n_obj)
    xl, xu = resolve_bounds(problem, encoding)

    span = xu - xl
    vmax = np.abs(span) * vmax_fraction
    vmax[vmax == 0.0] = 1.0

    # Initialize population
    initializer_cfg = config.get("initializer")
    X = initialize_population(pop_size, n_var, xl, xu, encoding, rng, problem, initializer=initializer_cfg)

    # Evaluate initial population
    eval_init = eval_backend.evaluate(X, problem)
    F = eval_init.F
    constraint_mode = config.get("constraint_mode", "feasibility")
    G = eval_init.G if constraint_mode != "none" else None

    # Initialize velocity
    velocity = rng.uniform(-vmax, vmax, size=X.shape)

    # Personal bests
    pbest_X = X.copy()
    pbest_F = F.copy()
    pbest_G = G.copy() if G is not None else None

    # Leader archive
    leader_archive = CrowdingDistanceArchive(archive_size, n_var, n_obj, X.dtype)
    archive_X, archive_F = leader_archive.update(X, F)

    # Genealogy tracking
    track_genealogy = bool(config.get("track_genealogy", False))
    genealogy_tracker, ids = setup_genealogy(pop_size, F, track_genealogy, "smpso")

    # Build operators
    mutation_op = build_mutation_operator(config, encoding, n_var, xl, xu)
    repair_op = build_repair_operator(config)

    # HV tracker
    hv_tracker = setup_hv_tracker(hv_config, kernel)

    # Create state
    state = SMPSOState(
        X=X,
        F=F,
        G=G,
        rng=rng,
        pop_size=pop_size,
        offspring_size=pop_size,
        constraint_mode=constraint_mode,
        n_eval=pop_size,
        generation=0,
        # Archive (leaders)
        archive_size=archive_size,
        archive_X=archive_X,
        archive_F=archive_F,
        archive_manager=leader_archive,
        # Termination
        hv_tracker=hv_tracker,
        # PSO state
        velocity=velocity,
        pbest_X=pbest_X,
        pbest_F=pbest_F,
        pbest_G=pbest_G,
        inertia=inertia,
        c1=c1,
        c2=c2,
        vmax=vmax,
        xl=xl,
        xu=xu,
        mutation_op=mutation_op,
        repair_op=repair_op,
        # Genealogy
        track_genealogy=track_genealogy,
        genealogy_tracker=genealogy_tracker,
        ids=ids,
    )

    return state, live_cb, eval_backend, max_eval, hv_tracker
