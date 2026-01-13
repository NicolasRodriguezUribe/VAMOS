"""SMS-EMOA core algorithm implementation.

S-Metric Selection Evolutionary Multiobjective Optimization Algorithm uses
hypervolume contribution for survival selection.

References:
    N. Beume, B. Naujoks, and M. Emmerich, "SMS-EMOA: Multiobjective Selection
    Based on Dominated Hypervolume," European Journal of Operational Research,
    vol. 181, no. 3, 2007.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from vamos.engine.algorithm.components.hooks import (
    finalize_genealogy,
    live_should_stop,
    track_offspring_genealogy,
)
from .helpers import (
    evaluate_population_with_constraints,
    survival_selection,
)
from .initialization import initialize_smsemoa_run
from .state import SMSEMOAState, build_smsemoa_result
from vamos.engine.algorithm.components.termination import HVTracker

if TYPE_CHECKING:
    from vamos.foundation.eval.backends import EvaluationBackend
    from vamos.foundation.kernel.backend import KernelBackend
    from vamos.foundation.problem.types import ProblemProtocol
from vamos.hooks.live_viz import LiveVisualization
from vamos.foundation.observer import RunContext


__all__ = ["SMSEMOA"]


class SMSEMOA:
    """S-Metric Selection Evolutionary Multiobjective Optimization Algorithm.

    SMS-EMOA uses hypervolume contribution for survival selection, removing
    solutions that contribute least to the hypervolume indicator each generation.
    This provides strong convergence toward the Pareto front.

    Parameters
    ----------
    config : dict
        Algorithm configuration with keys:
        - pop_size (int): Population size
        - crossover (tuple): Crossover operator config
        - mutation (tuple): Mutation operator config
        - selection (tuple): Selection config, e.g., ("tournament", {"pressure": 2})
        - reference_point (dict, optional): {"offset": float} or {"point": list}
        - external_archive_size (int, optional): Size of external archive
        - archive_type (str, optional): Archive type ("hypervolume" or "crowding")
        - hv_threshold (float, optional): HV threshold for early termination
        - hv_ref_point (list, optional): Reference point for HV computation
    kernel : KernelBackend
        Backend for vectorized operations.

    Examples
    --------
    Basic usage:

    >>> from vamos.engine.api import SMSEMOAConfig
    >>> config = SMSEMOAConfig().pop_size(100).crossover("sbx", prob=0.9).fixed()
    >>> smsemoa = SMSEMOA(config, kernel)
    >>> result = smsemoa.run(problem, ("n_eval", 10000), seed=42)

    Ask/tell interface:

    >>> smsemoa = SMSEMOA(config, kernel)
    >>> smsemoa.initialize(problem, ("n_eval", 10000), seed=42)
    >>> while not smsemoa.should_terminate():
    ...     X = smsemoa.ask()
    ...     F = evaluate(X)
    ...     smsemoa.tell(X, F)
    >>> result = smsemoa.result()
    """

    def __init__(self, config: dict[str, Any], kernel: "KernelBackend"):
        self.cfg = config
        self.kernel = kernel
        self._st: SMSEMOAState | None = None
        self._live_cb: "LiveVisualization | None" = None
        self._eval_strategy: "EvaluationBackend | None" = None
        self._max_eval: int = 0
        self._hv_tracker: HVTracker | None = None
        self._problem: "ProblemProtocol | None" = None

    # -------------------------------------------------------------------------
    # Main run method (batch mode)
    # -------------------------------------------------------------------------

    def run(
        self,
        problem: "ProblemProtocol",
        termination: tuple[str, Any],
        seed: int,
        eval_strategy: "EvaluationBackend | None" = None,
        live_viz: "LiveVisualization | None" = None,
    ) -> dict[str, Any]:
        """Run SMS-EMOA optimization loop.

        Parameters
        ----------
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
        dict
            Result dictionary with X, F, G, reference_point, archive data.
        """
        self._st, live_cb, eval_strategy, max_eval, hv_tracker = initialize_smsemoa_run(
            self.cfg, self.kernel, problem, termination, seed, eval_strategy, live_viz
        )
        self._problem = problem
        self._live_cb = live_cb
        self._eval_strategy = eval_strategy
        self._max_eval = max_eval
        self._hv_tracker = cast(HVTracker, hv_tracker)

        st = self._st
        ctx = RunContext(
            problem=problem,
            algorithm=self,
            config=self.cfg,
            algorithm_name="smsemoa",
            engine_name=str(self.kernel.name),
        )
        live_cb.on_start(ctx)

        hv_reached = False
        stop_requested = False
        while st.n_eval < max_eval and not stop_requested:
            # Generate and evaluate offspring
            X_child = self._generate_offspring(st)

            # Evaluate using backend or directly
            F_child, G_child = self._evaluate_offspring(problem, X_child, eval_strategy, st.constraint_mode)
            st.n_eval += X_child.shape[0]

            # Survival selection (one child at a time for SMS-EMOA)
            for i in range(X_child.shape[0]):
                survival_selection(
                    st,
                    X_child[i : i + 1],
                    F_child[i : i + 1],
                    G_child[i : i + 1] if G_child is not None else None,
                    self.kernel,
                )

            st.generation += 1

            # Update archive
            if st.archive_manager is not None:
                st.archive_X, st.archive_F = st.archive_manager.update(st.X, st.F)

            # Live callback
            live_cb.on_generation(st.generation, F=st.F, stats={"evals": st.n_eval})
            stop_requested = live_should_stop(live_cb)

            # Check HV threshold
            if hv_tracker is not None and hv_tracker.enabled and hv_tracker.reached(st.hv_points()):
                hv_reached = True
                break

        live_cb.on_end(final_F=st.F)
        result = build_smsemoa_result(st, hv_reached, kernel=self.kernel)
        finalize_genealogy(result, st, self.kernel)
        return result

    # -------------------------------------------------------------------------
    # Generation logic
    # -------------------------------------------------------------------------

    def _generate_offspring(self, st: SMSEMOAState) -> np.ndarray:
        """Generate offspring using parent selection and variation."""
        sel_cfg = self.cfg.get("selection", ("random", {}))
        if isinstance(sel_cfg, tuple):
            sel_method, _ = sel_cfg
        else:
            sel_method = sel_cfg

        if sel_method == "tournament":
            ranks, crowd = self.kernel.nsga2_ranking(st.F)
            parents_idx = self.kernel.tournament_selection(ranks, crowd, st.pressure, st.rng, n_parents=2)
        else:
            if st.X.shape[0] == 0:
                raise ValueError("Cannot select parents from an empty population.")
            parents_idx = st.rng.choice(st.X.shape[0], size=2, replace=True)

        parents = st.X[parents_idx]
        if parents.ndim == 2:
            parents = parents.reshape(1, 2, -1)

        # Apply crossover and mutation
        if st.crossover_fn is None or st.mutation_fn is None:
            raise RuntimeError("SMS-EMOA variation operators are not initialized.")
        offspring = st.crossover_fn(parents)
        child_vec = offspring.reshape(-1, st.X.shape[1])[0:1]  # first child as (1, n_var)
        child = st.mutation_fn(child_vec)

        # Track genealogy - store parent indices for later assignment
        if st.genealogy_tracker is not None:
            track_offspring_genealogy(
                st,
                parents_idx.reshape(-1),
                1,
                "sbx+pm",
                "smsemoa",
            )

        return child

    def _evaluate_offspring(
        self,
        problem: "ProblemProtocol",
        X: np.ndarray,
        eval_strategy: "EvaluationBackend",
        constraint_mode: str,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Evaluate offspring and compute constraints."""
        F, G = evaluate_population_with_constraints(problem, X)
        if constraint_mode == "none":
            G = None
        return F, G

    # -------------------------------------------------------------------------
    # Ask/Tell Interface
    # -------------------------------------------------------------------------

    def initialize(
        self,
        problem: "ProblemProtocol",
        termination: tuple[str, Any],
        seed: int,
        eval_strategy: "EvaluationBackend | None" = None,
        live_viz: "LiveVisualization | None" = None,
    ) -> None:
        """Initialize algorithm for ask/tell loop.

        Parameters
        ----------
        problem : ProblemProtocol
            Problem to optimize.
        termination : tuple
            Termination criterion.
        seed : int
            Random seed.
        eval_strategy : EvaluationBackend, optional
            Evaluation backend.
        live_viz : LiveVisualization, optional
            Live visualization callback.
        """
        self._st, self._live_cb, self._eval_strategy, self._max_eval, self._hv_tracker = initialize_smsemoa_run(
            self.cfg, self.kernel, problem, termination, seed, eval_strategy, live_viz
        )
        self._hv_tracker = cast(HVTracker, self._hv_tracker)
        self._problem = problem
        if self._st is not None:
            self._st.pending_offspring = None
        if self._live_cb is not None:
            ctx = RunContext(
                problem=problem,
                algorithm=self,
                config=self.cfg,
                algorithm_name="smsemoa",
                engine_name=str(self.kernel.name),
            )
            self._live_cb.on_start(ctx)

    def ask(self) -> np.ndarray:
        """Generate offspring for evaluation.

        Returns
        -------
        np.ndarray
            Offspring decision vectors to evaluate.

        Raises
        ------
        RuntimeError
            If algorithm not initialized or previous offspring not consumed.
        """
        if self._st is None:
            raise RuntimeError("Algorithm not initialized. Call initialize() first.")
        if self._st.pending_offspring is not None:
            raise RuntimeError("Previous offspring not yet consumed by tell().")

        offspring = self._generate_offspring(self._st)
        self._st.pending_offspring = offspring
        return offspring.copy()

    def tell(
        self,
        X: np.ndarray,
        F: np.ndarray,
        G: np.ndarray | None = None,
    ) -> None:
        """Receive evaluated offspring and update population.

        Parameters
        ----------
        X : np.ndarray
            Evaluated decision vectors.
        F : np.ndarray
            Objective values.
        G : np.ndarray, optional
            Constraint values.

        Raises
        ------
        RuntimeError
            If algorithm not initialized or no pending offspring.
        """
        if self._st is None:
            raise RuntimeError("Algorithm not initialized. Call initialize() first.")
        if self._st.pending_offspring is None:
            raise RuntimeError("No pending offspring. Call ask() first.")

        st = self._st
        st.pending_offspring = None

        # Survival selection for each child
        for i in range(X.shape[0]):
            G_i = G[i : i + 1] if G is not None else None
            survival_selection(st, X[i : i + 1], F[i : i + 1], G_i, self.kernel)

        st.n_eval += X.shape[0]
        st.generation += 1

        # Update archive
        if st.archive_manager is not None:
            st.archive_X, st.archive_F = st.archive_manager.update(st.X, st.F)

        # Live callback
        if self._live_cb is not None:
            self._live_cb.on_generation(st.generation, F=st.F)

        # Check HV tracker
        if st.hv_tracker is not None and st.hv_tracker.enabled:
            st.hv_tracker.reached(st.hv_points())

    def should_terminate(self) -> bool:
        """Check if termination criterion is met.

        Returns
        -------
        bool
            True if algorithm should stop.
        """
        if self._st is None:
            return True
        if self._st.n_eval >= self._max_eval:
            return True
        if self._st.hv_tracker is not None and self._st.hv_tracker.enabled and self._st.hv_tracker.reached(self._st.hv_points()):
            return True
        return False

    def result(self) -> dict[str, Any]:
        """Get current result.

        Returns
        -------
        dict
            Result dictionary with X, F, G, reference_point, archive data.

        Raises
        ------
        RuntimeError
            If algorithm not initialized.
        """
        if self._st is None:
            raise RuntimeError("Algorithm not initialized.")

        hv_reached = self._st.hv_tracker is not None and self._st.hv_tracker.enabled and self._st.hv_tracker.reached(self._st.hv_points())

        if self._live_cb is not None:
            self._live_cb.on_end(final_F=self._st.F)

        result = build_smsemoa_result(self._st, hv_reached, kernel=self.kernel)
        finalize_genealogy(result, self._st, self.kernel)
        return result

    @property
    def state(self) -> SMSEMOAState | None:
        """Access current algorithm state."""
        return self._st
