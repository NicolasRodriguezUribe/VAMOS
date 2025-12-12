# algorithm/nsgaii/core.py
"""
NSGA-II evolutionary algorithm core.

This module contains the main NSGAII class with the evolutionary loop (run/ask/tell).
- Setup logic: setup.py
- State and results: state.py
- Helper functions: helpers.py
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from vamos.algorithm.components.population import resolve_bounds
from vamos.algorithm.components.termination import HVTracker
from vamos.algorithm.components.variation import prepare_mutation_params
from vamos.algorithm.nsgaii.setup import (
    parse_termination,
    setup_population,
    setup_archive,
    setup_genealogy,
    setup_selection,
    setup_result_archive,
    build_operator_pool,
    resolve_archive_size,
)
from vamos.algorithm.nsgaii.state import (
    NSGAIIState,
    build_result,
    finalize_genealogy,
    compute_selection_metrics,
    track_offspring_genealogy,
    update_archives,
)
from vamos.algorithm.nsgaii.helpers import (
    build_mating_pool,
    feasible_nsga2_survival,
    match_ids,
    operator_success_stats,
    generation_contributions,
)
from vamos.operators.real import VariationWorkspace
from vamos.eval.backends import SerialEvalBackend, EvaluationBackend
from vamos.visualization.live_viz import LiveVisualization, NoOpLiveVisualization
from vamos.hyperheuristics.operator_selector import compute_reward
from vamos.kernel.backend import KernelBackend
from vamos.problem.types import ProblemProtocol

_logger = logging.getLogger(__name__)

# Backward-compat aliases for modules still importing the old private helpers
_build_mating_pool = build_mating_pool
_feasible_nsga2_survival = feasible_nsga2_survival
_match_ids = match_ids
_operator_success_stats = operator_success_stats
_generation_contributions = generation_contributions


class NSGAII:
    """
    Vectorized/SOA-style NSGA-II evolutionary core.
    Individuals are represented as array rows (X, F) without per-object instances.
    """

    def __init__(self, config: dict[str, Any], kernel: KernelBackend) -> None:
        self.cfg = config
        self.kernel = kernel
        self._st: NSGAIIState | None = None

    def run(
        self,
        problem: ProblemProtocol,
        termination: tuple[str, Any],
        seed: int,
        eval_backend: EvaluationBackend | None = None,
        live_viz: LiveVisualization | None = None,
    ) -> dict[str, Any]:
        """Run the NSGA-II algorithm."""
        live_cb, eval_backend, max_eval, n_eval, hv_tracker = self._initialize_run(
            problem, termination, seed, eval_backend, live_viz
        )
        st = self._st
        assert st is not None, "State not initialized"

        generation = 0
        live_cb.on_generation(generation, F=st.F)
        hv_reached = hv_tracker.enabled and hv_tracker.reached(st.hv_points_fn())

        while n_eval < max_eval and not hv_reached:
            st.generation = generation
            X_off = self.ask()
            eval_off = eval_backend.evaluate(X_off, problem)
            hv_reached = self.tell(eval_off, st.pop_size)
            n_eval += X_off.shape[0]

            if hv_tracker.enabled and hv_tracker.reached(st.hv_points_fn()):
                hv_reached = True
                break

            generation += 1
            st.generation = generation
            self._notify_generation(live_cb, generation, st.F)

        result = build_result(st, n_eval, hv_reached)
        live_cb.on_end(final_F=st.F)
        finalize_genealogy(result, st, self.kernel)
        return result

    def _notify_generation(self, live_cb: LiveVisualization, generation: int, F: np.ndarray) -> None:
        """Notify live visualization of generation progress."""
        try:
            ranks, _ = self.kernel.nsga2_ranking(F)
            nd_mask = ranks == ranks.min(initial=0)
            live_cb.on_generation(generation, F=F[nd_mask])
        except (ValueError, IndexError) as exc:
            _logger.debug("Failed to compute non-dominated front for viz: %s", exc)
            live_cb.on_generation(generation, F=F)

    def _initialize_run(
        self,
        problem: ProblemProtocol,
        termination: tuple[str, Any],
        seed: int,
        eval_backend: EvaluationBackend | None,
        live_viz: LiveVisualization | None,
    ) -> tuple[LiveVisualization, EvaluationBackend, int, int, HVTracker]:
        """Initialize algorithm state for a run."""
        max_eval, hv_config = parse_termination(termination)

        if eval_backend is None:
            eval_backend = SerialEvalBackend()
        live_cb = live_viz or NoOpLiveVisualization()
        rng = np.random.default_rng(seed)

        pop_size = int(self.cfg["pop_size"])
        offspring_size = int(self.cfg.get("offspring_size") or pop_size)
        if offspring_size <= 0:
            raise ValueError("offspring size must be positive.")

        constraint_mode = self.cfg.get("constraint_mode", "feasibility")
        initializer_cfg = self.cfg.get("initializer")
        X, F, G, n_eval = setup_population(
            problem, eval_backend, rng, pop_size, constraint_mode, initializer_cfg
        )

        encoding = getattr(problem, "encoding", "continuous")
        n_var = problem.n_var
        xl, xu = resolve_bounds(problem, encoding)

        live_cb.on_start(problem=problem, algorithm=self, config=self.cfg)
        hv_tracker = HVTracker(hv_config, self.kernel)

        archive_size = resolve_archive_size(self.cfg)
        archive_X, archive_F, archive_manager, archive_via_kernel = setup_archive(
            self.kernel, X, F, n_var, problem.n_obj, X.dtype, archive_size
        )

        track_genealogy = bool(self.cfg.get("track_genealogy", False))
        genealogy_tracker, ids = setup_genealogy(pop_size, F, track_genealogy)

        sel_method, sel_params = self.cfg["selection"]
        sel_method, pressure = setup_selection(sel_method, sel_params)

        cross_method, cross_params = self.cfg["crossover"]
        cross_method = cross_method.lower()
        cross_params = dict(cross_params)

        mut_method, mut_params = self.cfg["mutation"]
        mut_method = mut_method.lower()
        mut_factor = self.cfg.get("mutation_prob_factor")
        mut_params = prepare_mutation_params(mut_params, encoding, n_var, prob_factor=mut_factor)

        variation_workspace = VariationWorkspace()
        operator_pool, op_selector, indicator_eval = build_operator_pool(
            self.cfg, encoding, cross_method, cross_params, mut_method, mut_params,
            n_var, xl, xu, variation_workspace, problem, mut_factor,
        )

        result_mode = self.cfg.get("result_mode", "population")
        archive_type = self.cfg.get("archive_type", "hypervolume")
        result_archive = setup_result_archive(
            result_mode, archive_type, archive_size, n_var, problem.n_obj, X.dtype
        )

        self._st = NSGAIIState(
            X=X, F=F, G=G, rng=rng,
            variation=operator_pool[0], operator_pool=operator_pool,
            variation_workspace=variation_workspace,
            op_selector=op_selector, indicator_eval=indicator_eval,
            sel_method=sel_method, pressure=pressure,
            pop_size=pop_size, offspring_size=offspring_size,
            constraint_mode=constraint_mode,
            archive_size=archive_size, archive_X=archive_X, archive_F=archive_F,
            archive_manager=archive_manager, archive_via_kernel=archive_via_kernel,
            result_archive=result_archive, result_mode=result_mode,
            hv_tracker=hv_tracker,
            track_genealogy=track_genealogy, genealogy_tracker=genealogy_tracker, ids=ids,
        )
        # Legacy dict interface for backward compat
        self._state = self._st.__dict__

        return live_cb, eval_backend, max_eval, n_eval, hv_tracker

    def ask(self) -> np.ndarray:
        """Generate offspring from the current state (minimal ask/tell support)."""
        st = self._st
        if st is None:
            raise RuntimeError("ask() called before initialization.")

        if st.op_selector is not None:
            idx = st.op_selector.select_operator()
            st.variation = st.operator_pool[idx]
            st.last_operator_idx = idx

        ranks, crowding = compute_selection_metrics(self.kernel, st.F, st.G, st.constraint_mode)
        parents_per_group = st.variation.parents_per_group
        children_per_group = st.variation.children_per_group
        parent_count = int(np.ceil(st.offspring_size / children_per_group) * parents_per_group)

        mating_pairs = build_mating_pool(
            self.kernel, ranks, crowding, st.pressure, st.rng,
            parent_count, parents_per_group, st.sel_method,
        )
        parent_idx = mating_pairs.reshape(-1)
        X_parents = st.variation.gather_parents(st.X, parent_idx)
        X_off = st.variation.produce_offspring(X_parents, st.rng)

        if X_off.shape[0] > st.offspring_size:
            X_off = X_off[:st.offspring_size]
        st.pending_offspring = X_off

        track_offspring_genealogy(st, parent_idx, X_off.shape[0])
        return X_off

    def tell(self, eval_result: Any, pop_size: int) -> bool:
        """Consume evaluated offspring and update state. Returns hv_reached flag."""
        st = self._st
        if st is None:
            raise RuntimeError("tell() called before initialization.")

        X_off = st.pending_offspring
        st.pending_offspring = None
        if X_off is None:
            raise ValueError("tell() called without a pending ask().")

        F_off = eval_result.F
        G_off = eval_result.G if st.constraint_mode != "none" else None
        hv_before = self._compute_hv_before(st)

        # Combine IDs for genealogy
        combined_X = np.vstack([st.X, X_off])
        combined_ids = self._combine_ids(st)

        # Survival selection
        if st.G is None or G_off is None or st.constraint_mode == "none":
            new_X, new_F = self.kernel.nsga2_survival(st.X, st.F, X_off, F_off, pop_size)
            new_G = None
        else:
            new_X, new_F, new_G = feasible_nsga2_survival(
                self.kernel, st.X, st.F, st.G, X_off, F_off, G_off, pop_size
            )

        if combined_ids is not None:
            st.ids = match_ids(new_X, combined_X, combined_ids)

        st.X, st.F, st.G = new_X, new_F, new_G
        st.pending_offspring_ids = None

        update_archives(st, self.kernel)

        hv_reached = st.hv_tracker.enabled and st.hv_tracker.reached(st.hv_points_fn())
        self._update_operator_selector(st, hv_before)

        return hv_reached

    def _compute_hv_before(self, st: NSGAIIState) -> float | None:
        """Compute HV before survival for adaptive operator selection."""
        if st.indicator_eval is None:
            return None
        try:
            return st.indicator_eval.compute(st.hv_points_fn())
        except (ValueError, TypeError, RuntimeError) as exc:
            _logger.debug("Failed to compute HV before: %s", exc)
            return None

    def _combine_ids(self, st: NSGAIIState) -> np.ndarray | None:
        """Combine parent and offspring IDs for genealogy tracking."""
        if not st.track_genealogy:
            return None
        current_ids = st.ids if st.ids is not None else np.array([], dtype=int)
        pending_ids = st.pending_offspring_ids if st.pending_offspring_ids is not None else np.array([], dtype=int)
        return np.concatenate([current_ids, pending_ids])

    def _update_operator_selector(self, st: NSGAIIState, hv_before: float | None) -> None:
        """Update adaptive operator selector with reward."""
        if st.op_selector is None or hv_before is None:
            return
        try:
            hv_after = st.indicator_eval.compute(st.hv_points_fn())
            reward = compute_reward(hv_before, hv_after, st.indicator_eval.mode)
            st.op_selector.update(st.last_operator_idx, reward)
        except (ValueError, TypeError, RuntimeError) as exc:
            _logger.debug("Failed to compute operator reward: %s", exc)
