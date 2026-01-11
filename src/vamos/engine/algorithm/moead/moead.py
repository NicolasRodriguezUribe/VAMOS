# algorithm/moead/core.py
"""
MOEA/D evolutionary algorithm core.

This module contains the main MOEAD class with the evolutionary loop (run/ask/tell).
- Setup logic: setup.py
- Operator building: operators.py
- State and results: state.py
- Helper functions: helpers.py

References:
    Q. Zhang and H. Li, "MOEA/D: A Multiobjective Evolutionary Algorithm Based on
    Decomposition," IEEE Trans. Evolutionary Computation, vol. 11, no. 6, 2007.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.engine.algorithm.components.archives import update_archive
from vamos.engine.algorithm.components.hooks import (
    finalize_genealogy,
    live_should_stop,
    notify_generation,
    track_offspring_genealogy,
)
from vamos.foundation.constraints.utils import compute_violation

from .helpers import update_neighborhood
from .initialization import initialize_moead_run
from .state import MOEADState, build_moead_result

if TYPE_CHECKING:
    from vamos.foundation.eval.backends import EvaluationBackend
    from vamos.foundation.kernel.backend import KernelBackend
    from vamos.foundation.problem.types import ProblemProtocol
from vamos.hooks.live_viz import LiveVisualization


class MOEAD:
    """
    Multi-Objective Evolutionary Algorithm based on Decomposition.

    MOEA/D decomposes a multi-objective problem into scalar subproblems using
    weight vectors and optimizes them collaboratively via neighborhood-based
    mating and replacement.

    Parameters
    ----------
    config : dict
        Algorithm configuration with keys:
        - pop_size (int): Population size (should match number of weight vectors)
        - crossover (tuple): Crossover operator config, e.g., ("sbx", {"prob": 0.9})
        - mutation (tuple): Mutation operator config, e.g., ("pm", {"prob": "1/n"})
        - weight_vectors (dict, optional): {"path": str, "divisions": int}
        - neighbor_size (int, optional): T parameter (default: 20)
        - delta (float, optional): Neighborhood selection probability (default: 0.9)
        - replace_limit (int, optional): Max replacements per offspring (default: 2)
        - aggregation (tuple, optional): ("pbi", {"theta": 5.0}) or similar
        - constraint_mode (str, optional): "feasibility" or "none"
        - archive (dict, optional): {"size": int, "type": str}
    kernel : KernelBackend
        Backend for vectorized operations.

    Attributes
    ----------
    cfg : dict
        Stored configuration.
    kernel : KernelBackend
        Kernel backend instance.

    Examples
    --------
    >>> from vamos.engine.api import MOEADConfig
    >>> config = MOEADConfig().pop_size(100).crossover("sbx", prob=0.9).fixed()
    >>> moead = MOEAD(config, kernel)
    >>> result = moead.run(problem, ("n_eval", 10000), seed=42)

    Using ask/tell for external evaluation:
    >>> moead._initialize_run(problem, ("n_eval", 10000), seed=42)
    >>> while moead._st.n_eval < 10000:
    ...     X_off = moead.ask()
    ...     F_off = my_external_evaluator(X_off)
    ...     moead.tell(EvalResult(F=F_off, G=None))
    """

    def __init__(self, config: dict[str, Any], kernel: "KernelBackend") -> None:
        self.cfg = config
        self.kernel = kernel
        self._st: MOEADState | None = None

    def run(
        self,
        problem: "ProblemProtocol",
        termination: tuple[str, Any],
        seed: int,
        eval_strategy: "EvaluationBackend | None" = None,
        live_viz: "LiveVisualization | None" = None,
    ) -> dict[str, Any]:
        """
        Run the MOEA/D algorithm.

        Parameters
        ----------
        problem : ProblemProtocol
            The optimization problem to solve.
        termination : tuple[str, Any]
            Termination criterion: ("n_eval", N) or ("hv", {...}).
        seed : int
            Random seed for reproducibility.
        eval_strategy : EvaluationBackend | None
            Optional evaluation backend for parallel evaluation.
        live_viz : LiveVisualization | None
            Optional live visualization callback.

        Returns
        -------
        dict[str, Any]
            Result dictionary with X, F, weights, evaluations, and optional archive.
        """
        live_cb, eval_strategy, max_eval, hv_tracker = self._initialize_run(problem, termination, seed, eval_strategy, live_viz)
        st = self._st
        assert st is not None, "State not initialized"

        generation = 0
        live_cb.on_generation(generation, F=st.F, stats={"evals": st.n_eval})
        stop_requested = live_should_stop(live_cb)
        hv_reached = hv_tracker.enabled and hv_tracker.reached(st.hv_points())

        while st.n_eval < max_eval and not hv_reached and not stop_requested:
            st.generation = generation
            X_off = self.ask()
            eval_result = eval_strategy.evaluate(X_off, problem)
            hv_reached = self.tell(eval_result, problem)

            if hv_tracker.enabled and hv_tracker.reached(st.hv_points()):
                hv_reached = True
                break

            generation += 1
            st.generation = generation
            stop_requested = notify_generation(live_cb, self.kernel, generation, st.F, stats={"evals": st.n_eval})

        result = build_moead_result(st, hv_reached, kernel=self.kernel)
        finalize_genealogy(result, st, self.kernel)
        live_cb.on_end(final_F=st.F)
        return result

    def _initialize_run(
        self,
        problem: "ProblemProtocol",
        termination: tuple[str, Any],
        seed: int,
        eval_strategy: "EvaluationBackend | None" = None,
        live_viz: "LiveVisualization | None" = None,
    ) -> tuple[Any, Any, int, Any]:
        """Initialize algorithm state for a run."""
        self._st, live_cb, eval_strategy, max_eval, hv_tracker = initialize_moead_run(
            self.cfg, self.kernel, problem, termination, seed, eval_strategy, live_viz
        )
        return live_cb, eval_strategy, max_eval, hv_tracker

    def ask(self) -> np.ndarray:
        """
        Generate offspring solutions to be evaluated.

        Returns
        -------
        np.ndarray
            Offspring decision variables to evaluate, shape (batch_size, n_var).

        Raises
        ------
        RuntimeError
            If called before initialization.
        """
        st = self._st
        if st is None:
            raise RuntimeError("ask() called before initialization.")

        pop_size = st.pop_size
        batch_size = int(st.batch_size)
        if batch_size <= 0:
            raise ValueError("MOEA/D batch_size must be positive.")
        if batch_size > pop_size:
            batch_size = pop_size
        active = self._next_active_indices(st, batch_size)
        use_neighbors = st.rng.random(batch_size) < st.delta

        # Select parent pairs
        all_indices = np.arange(pop_size)
        parent_pairs = np.empty((batch_size, 2), dtype=int)

        for pos, i in enumerate(active):
            mating_pool = st.neighbors[i] if use_neighbors[pos] else all_indices
            if mating_pool.size < 2:
                mating_pool = all_indices
            parent_pairs[pos] = st.rng.choice(mating_pool, size=2, replace=False)

        # Generate offspring
        n_var = st.X.shape[1]
        cross_method = str(self.cfg.get("crossover", ("sbx", {}))[0]).lower()
        if cross_method in {"de", "differential", "differential_evolution"}:
            parents = np.empty((batch_size, 3, n_var), dtype=st.X.dtype)
            parents[:, 0, :] = st.X[parent_pairs[:, 0]]
            parents[:, 1, :] = st.X[parent_pairs[:, 1]]
            parents[:, 2, :] = st.X[active]
            offspring = st.crossover_fn(parents)
            children = offspring[:, 0, :].copy()
            parents_flat = np.column_stack([parent_pairs, active]).reshape(-1)
        else:
            parents_flat = parent_pairs.reshape(-1)
            parents = st.X[parents_flat].reshape(batch_size, 2, n_var)
            offspring = st.crossover_fn(parents)
            children = offspring[:, 0, :].copy()

        children = st.mutation_fn(children)

        # Store pending info for tell()
        st.pending_offspring = children
        st.pending_active_indices = active
        st.pending_parent_pairs = parent_pairs
        st.pending_use_neighbors = use_neighbors

        # Track genealogy
        op_name = "de+pm" if cross_method in {"de", "differential", "differential_evolution"} else "sbx+pm"
        track_offspring_genealogy(st, parents_flat, children.shape[0], op_name, "moead")

        return children

    def tell(self, eval_result: Any, problem: "ProblemProtocol | None" = None) -> bool:
        """
        Receive evaluated offspring and update algorithm state.

        Parameters
        ----------
        eval_result : Any
            Evaluation result with F (objectives) and optionally G (constraints).
        problem : ProblemProtocol | None
            Problem instance (optional, for constraint evaluation).

        Returns
        -------
        bool
            True if HV threshold reached.

        Raises
        ------
        RuntimeError
            If called before initialization or without pending ask().
        """
        st = self._st
        if st is None:
            raise RuntimeError("tell() called before initialization.")

        children = st.pending_offspring
        active = st.pending_active_indices
        use_neighbors = st.pending_use_neighbors

        if children is None or active is None or use_neighbors is None:
            raise ValueError("tell() called without a pending ask().")

        F_child = eval_result.F
        G_child = eval_result.G if st.constraint_mode != "none" else None
        batch_size = children.shape[0]
        st.n_eval += batch_size

        # Clear pending
        st.pending_offspring = None
        st.pending_active_indices = None
        st.pending_parent_pairs = None
        st.pending_use_neighbors = None

        pop_size = st.pop_size

        # Update ideal point
        st.ideal = np.minimum(st.ideal, F_child.min(axis=0))

        # Update neighborhoods
        for pos, i in enumerate(active):
            child = children[pos]
            child_f = F_child[pos]
            child_g = G_child[pos] if G_child is not None else None
            cv_penalty = compute_violation(G_child)[pos] if G_child is not None else 0.0
            if use_neighbors[pos]:
                candidate_order = st.neighbors[i]
            else:
                candidate_order = st.rng.permutation(pop_size)

            update_neighborhood(
                st=st,
                idx=i,
                child=child,
                child_f=child_f,
                child_g=child_g,
                cv_penalty=cv_penalty,
                candidate_order=candidate_order,
            )

        # Update archive
        update_archive(st)

        # Check HV termination
        hv_reached = st.hv_tracker is not None and st.hv_tracker.enabled and st.hv_tracker.reached(st.hv_points())
        return hv_reached

    @staticmethod
    def _next_active_indices(st: MOEADState, batch_size: int) -> np.ndarray:
        """Return next subproblem indices using a rolling permutation."""
        if st.subproblem_order.size != st.pop_size:
            st.subproblem_order = st.rng.permutation(st.pop_size).astype(int, copy=False)
            st.subproblem_cursor = 0

        active = np.empty(batch_size, dtype=int)
        for i in range(batch_size):
            if st.subproblem_cursor >= st.pop_size:
                st.subproblem_order = st.rng.permutation(st.pop_size).astype(int, copy=False)
                st.subproblem_cursor = 0
            active[i] = int(st.subproblem_order[st.subproblem_cursor])
            st.subproblem_cursor += 1
        return active


__all__ = ["MOEAD"]
