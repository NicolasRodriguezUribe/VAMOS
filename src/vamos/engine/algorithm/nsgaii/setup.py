"""
Initialization helpers for NSGA-II.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING, cast
from collections.abc import Mapping

import numpy as np

from vamos.engine.algorithm.components.population import resolve_bounds
from vamos.engine.algorithm.components.termination import HVTracker
from vamos.engine.algorithm.components.variation import prepare_mutation_params
from vamos.foundation.encoding import normalize_encoding
from vamos.foundation.checkpoint import restore_rng
from vamos.foundation.eval.backends import EvaluationBackend, SerialEvalBackend
from vamos.foundation.observer import RunContext
from vamos.hooks.live_viz import LiveVisualization, NoOpLiveVisualization
from vamos.operators.impl.real import VariationWorkspace
from vamos.operators.policies.nsgaii import build_operator_pool

from .helpers import fronts_from_ranks
from .injection import ImmigrationManager
from .initialization import (
    parse_termination,
    resolve_archive_size,
    setup_archive,
    setup_genealogy,
    setup_population,
    setup_result_archive,
    setup_selection,
)
from .state import NSGAIIState

if TYPE_CHECKING:
    from .nsgaii import NSGAII


def _resolve_archive_settings(cfg: dict[str, Any], *, pop_size: int) -> tuple[int | None, bool]:
    archive_cfg = cfg.get("archive") or cfg.get("external_archive") or {}
    unbounded = bool(archive_cfg.get("unbounded", False)) if isinstance(archive_cfg, dict) else False
    size = resolve_archive_size(cfg)
    if unbounded and not size:
        size = pop_size
    return size, unbounded


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
    steady_state = bool(algo.cfg.get("steady_state", False))
    raw_offspring_size = algo.cfg.get("offspring_size")
    replacement_size = algo.cfg.get("replacement_size")

    if steady_state:
        if replacement_size is None:
            if raw_offspring_size is not None and int(raw_offspring_size) != pop_size:
                replacement_size = raw_offspring_size
            else:
                replacement_size = 1
        replacement_size = int(replacement_size)
        if replacement_size <= 0:
            raise ValueError("replacement size must be positive.")
        if replacement_size > pop_size:
            raise ValueError("replacement size must be <= population size.")
        offspring_size = replacement_size
    else:
        offspring_size = int(raw_offspring_size or pop_size)
        if offspring_size <= 0:
            raise ValueError("offspring size must be positive.")
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

    incremental_enabled = bool(steady_state and replacement_size == 1 and constraint_mode == "none" and G is None)
    if incremental_enabled and getattr(algo.kernel, "name", "") == "jax" and getattr(algo.kernel, "_strict_ranking", True) is False:
        incremental_enabled = False

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

    archive_size, archive_unbounded = _resolve_archive_settings(algo.cfg, pop_size=pop_size)
    archive_X, archive_F, archive_manager, archive_via_kernel = setup_archive(
        algo.kernel,
        X,
        F,
        n_var,
        problem.n_obj,
        X.dtype,
        archive_size,
        unbounded=archive_unbounded,
    )

    if checkpoint_archive_X is not None and checkpoint_archive_F is not None and archive_size:
        try:
            if archive_manager is not None:
                archive_X, archive_F = archive_manager.update(checkpoint_archive_X, checkpoint_archive_F)
            elif archive_via_kernel:
                archive_X, archive_F = algo.kernel.update_archive(
                    None,
                    None,
                    checkpoint_archive_X,
                    checkpoint_archive_F,
                    archive_size,
                )
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
    operator_pool, aos_controller = build_operator_pool(
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

    result_mode = algo.cfg.get("result_mode", "non_dominated")
    archive_type = algo.cfg.get("archive_type", "hypervolume")
    result_archive = setup_result_archive(
        result_mode,
        archive_type,
        archive_size,
        n_var,
        problem.n_obj,
        X.dtype,
        unbounded=archive_unbounded,
    )
    if result_archive is not None and checkpoint is not None:
        try:
            source_X = checkpoint_archive_X if checkpoint_archive_X is not None else X
            source_F = checkpoint_archive_F if checkpoint_archive_F is not None else F
            result_archive.update(source_X, source_F)
        except Exception:  # pragma: no cover - defensive
            pass

    immigration_cfg = algo.cfg.get("immigration")
    immigration_manager = None
    if isinstance(immigration_cfg, Mapping):
        immigration_manager = ImmigrationManager(immigration_cfg)

    parent_selection_filter = algo.cfg.get("parent_selection_filter")
    live_callback_mode = str(algo.cfg.get("live_callback_mode", "nd_only")).lower()
    if live_callback_mode not in {"nd_only", "population", "population_archive"}:
        raise ValueError(
            "live_callback_mode must be one of: nd_only, population, population_archive"
        )
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
        steady_state=steady_state,
        constraint_mode=constraint_mode,
        archive_size=archive_size,
        archive_X=archive_X,
        archive_F=archive_F,
        archive_manager=archive_manager,
        archive_via_kernel=archive_via_kernel,
        result_archive=result_archive,
        result_mode=result_mode,
        hv_tracker=hv_tracker,
        track_genealogy=track_genealogy,
        genealogy_tracker=genealogy_tracker,
        ids=ids,
        aos_controller=aos_controller,
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
