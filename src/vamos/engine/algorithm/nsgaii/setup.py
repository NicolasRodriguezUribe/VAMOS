"""
Initialization helpers for NSGA-II.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from vamos.engine.algorithm.components.population import resolve_bounds
from vamos.engine.algorithm.components.termination import HVTracker
from vamos.engine.algorithm.components.variation import prepare_mutation_params
from vamos.engine.archive.factory import resolve_external_archive, setup_archive
from vamos.engine.hooks.live_viz import LiveVisualization, NoOpLiveVisualization
from vamos.engine.operators.impl.real import VariationWorkspace
from vamos.engine.operators.policies.nsgaii import build_operator_pool
from vamos.foundation.checkpoint import restore_rng
from vamos.foundation.encoding import normalize_encoding
from vamos.foundation.eval.backends import EvaluationBackend, SerialEvalBackend
from vamos.foundation.observer import RunContext

from .helpers import fronts_from_ranks
from .initialization import (
    parse_termination,
    setup_genealogy,
    setup_population,
    setup_selection,
)
from .injection import ImmigrationManager
from .state import NSGAIIState

if TYPE_CHECKING:
    from vamos.engine.archive import ExternalArchiveConfig

    from .nsgaii import NSGAII


def _resolve_archive_settings(cfg: dict[str, Any]) -> ExternalArchiveConfig | None:
    return resolve_external_archive(cfg)


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def initialize_run(
    algo: NSGAII,
    problem: Any,
    termination: tuple[str, Any],
    seed: int,
    eval_strategy: EvaluationBackend | None,
    live_viz: LiveVisualization | None,
    checkpoint: Mapping[str, Any] | None = None,
) -> tuple[LiveVisualization, EvaluationBackend, int, int, HVTracker]:
    max_eval, hv_config = parse_termination(termination)

    if eval_strategy is None:
        eval_strategy = SerialEvalBackend()
    live_cb = live_viz or NoOpLiveVisualization()
    rng = np.random.default_rng(seed)

    pop_size = int(algo.cfg["pop_size"])
    offspring_size = int(algo.cfg.get("offspring_size") or pop_size)
    if offspring_size <= 0:
        raise ValueError("offspring size must be positive.")
    incremental_mode = offspring_size < pop_size
    replacement_size = 1

    constraint_mode = algo.cfg.get("constraint_mode", "feasibility")
    initializer_cfg = algo.cfg.get("initializer")
    X: np.ndarray
    F: np.ndarray
    G: np.ndarray | None
    n_eval: int
    generation = 0
    step = 0
    replacements = 0

    checkpoint_archive_X: np.ndarray | None = None
    checkpoint_archive_F: np.ndarray | None = None

    if checkpoint is not None:
        try:
            X = np.asarray(checkpoint["X"])
            F = np.asarray(checkpoint["F"])
            if X.ndim != 2 or F.ndim != 2:
                raise ValueError("Checkpoint X and F must be 2D arrays.")
            if X.shape[0] != pop_size:
                raise ValueError(f"Checkpoint pop_size={X.shape[0]} does not match config pop_size={pop_size}.")
            if F.shape[0] != X.shape[0]:
                raise ValueError("Checkpoint F row count must match X.")

            G_raw = checkpoint.get("G")
            G = np.asarray(G_raw) if G_raw is not None else None
            if constraint_mode == "none":
                G = None

            n_eval = int(checkpoint.get("n_eval", X.shape[0]))
            if n_eval < X.shape[0]:
                n_eval = X.shape[0]

            rng_state = checkpoint.get("rng_state")
            if rng_state is not None:
                try:
                    restore_rng(rng, cast(dict[str, Any], rng_state))
                except Exception as exc:  # pragma: no cover - defensive
                    _logger().warning("Failed to restore RNG state from checkpoint: %s", exc)

            generation = int(checkpoint.get("generation", 0))
            extra = checkpoint.get("extra", {})
            if isinstance(extra, Mapping):
                step = int(extra.get("step", 0))
                replacements = int(extra.get("replacements", max(0, n_eval - X.shape[0])))
            else:
                replacements = max(0, n_eval - X.shape[0])

            archive_x_raw = checkpoint.get("archive_X")
            archive_f_raw = checkpoint.get("archive_F")
            if archive_x_raw is not None and archive_f_raw is not None:
                checkpoint_archive_X = np.asarray(archive_x_raw)
                checkpoint_archive_F = np.asarray(archive_f_raw)
        except Exception as exc:
            _logger().warning("Invalid checkpoint provided, reinitializing population: %s", exc)
            X, F, G, n_eval = setup_population(problem, eval_strategy, rng, pop_size, constraint_mode, initializer_cfg)
            generation = 0
            step = 0
            replacements = 0
    else:
        X, F, G, n_eval = setup_population(problem, eval_strategy, rng, pop_size, constraint_mode, initializer_cfg)

    incremental_enabled = bool(incremental_mode and replacement_size == 1 and G is None)

    ranks = crowding = None
    fronts = None
    if incremental_enabled:
        ranks, crowding = algo.kernel.nsga2_ranking(F)
        fronts = fronts_from_ranks(ranks)

    encoding = normalize_encoding(getattr(problem, "encoding", "real"))
    n_var = problem.n_var
    xl, xu = resolve_bounds(problem, encoding)

    ctx = RunContext(
        problem=problem,
        algorithm=algo,
        config=algo.cfg,
        algorithm_name="nsgaii",
        engine_name=str(algo.kernel.name),
    )
    live_cb.on_start(ctx)
    hv_tracker = HVTracker(hv_config, algo.kernel)

    ext_cfg = _resolve_archive_settings(algo.cfg)
    archive_X, archive_F, archive_manager = setup_archive(
        algo.kernel,
        X,
        F,
        n_var,
        problem.n_obj,
        X.dtype,
        ext_cfg,
        G,
    )

    if checkpoint_archive_X is not None and checkpoint_archive_F is not None and ext_cfg is not None:
        try:
            if archive_manager is not None:
                archive_X, archive_F = archive_manager.update(checkpoint_archive_X, checkpoint_archive_F)
            else:
                archive_X = checkpoint_archive_X
                archive_F = checkpoint_archive_F
        except Exception as exc:  # pragma: no cover - defensive
            _logger().warning("Failed to restore archive from checkpoint: %s", exc)

    track_genealogy = bool(algo.cfg.get("track_genealogy", False))
    genealogy_tracker, ids = setup_genealogy(pop_size, F, track_genealogy)

    sel_method, sel_params = algo.cfg["selection"]
    sel_method, pressure = setup_selection(sel_method, sel_params)

    cross_method, cross_params = algo.cfg["crossover"]
    cross_method = cross_method.lower()
    cross_params = dict(cross_params)

    mut_method, mut_params = algo.cfg["mutation"]
    mut_method = mut_method.lower()
    mut_factor = algo.cfg.get("mutation_prob_factor")
    mut_params = prepare_mutation_params(mut_params, encoding, n_var, prob_factor=mut_factor)

    variation_workspace = VariationWorkspace()
    operator_pool = build_operator_pool(
        algo.cfg,
        encoding,
        cross_method,
        cross_params,
        mut_method,
        mut_params,
        n_var,
        xl,
        xu,
        variation_workspace,
        problem,
        mut_factor,
    )

    result_mode = str(algo.cfg.get("result_mode", "non_dominated")).strip().lower()
    if result_mode not in {"non_dominated", "population"}:
        raise ValueError("result_mode must be one of: non_dominated, population")
    result_archive = None
    if ext_cfg is not None and ext_cfg.capacity is not None and archive_manager is not None:
        result_archive = cast(Any, archive_manager)

    immigration_cfg = algo.cfg.get("immigration")
    immigration_manager = None
    if isinstance(immigration_cfg, Mapping):
        immigration_manager = ImmigrationManager(immigration_cfg)

    parent_selection_filter = algo.cfg.get("parent_selection_filter")
    live_callback_mode = str(algo.cfg.get("live_callback_mode", "nd_only")).lower()
    if live_callback_mode not in {"nd_only", "population", "population_archive"}:
        raise ValueError("live_callback_mode must be one of: nd_only, population, population_archive")
    generation_callback = algo.cfg.get("generation_callback")
    generation_callback_copy = bool(algo.cfg.get("generation_callback_copy", True))

    algo._st = NSGAIIState(
        X=X,
        F=F,
        G=G,
        rng=rng,
        variation=operator_pool[0],
        operator_pool=operator_pool,
        variation_workspace=variation_workspace,
        sel_method=sel_method,
        pressure=pressure,
        pop_size=pop_size,
        offspring_size=offspring_size,
        replacement_size=replacement_size,
        incremental_mode=incremental_mode,
        constraint_mode=constraint_mode,
        archive_size=ext_cfg.capacity if ext_cfg else None,
        archive_X=archive_X,
        archive_F=archive_F,
        archive_manager=archive_manager,
        result_archive=result_archive,
        result_mode=result_mode,
        hv_tracker=hv_tracker,
        track_genealogy=track_genealogy,
        genealogy_tracker=genealogy_tracker,
        ids=ids,
        fronts=fronts,
        ranks=ranks,
        crowding=crowding,
        incremental_enabled=incremental_enabled,
        generation=generation,
        step=step,
        replacements=replacements,
        immigration_manager=immigration_manager,
        parent_selection_filter=parent_selection_filter,
        live_callback_mode=live_callback_mode,
        generation_callback=generation_callback,
        generation_callback_copy=generation_callback_copy,
    )
    return live_cb, eval_strategy, max_eval, n_eval, hv_tracker


__all__ = ["initialize_run"]
