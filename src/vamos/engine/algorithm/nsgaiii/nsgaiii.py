"""NSGA-III core algorithm implementation.

Non-dominated Sorting Genetic Algorithm III uses reference points for
many-objective optimization (3+ objectives).

References:
    K. Deb and H. Jain, "An Evolutionary Many-Objective Optimization Algorithm
    Using Reference-Point-Based Nondominated Sorting Approach, Part I: Solving
    Problems With Box Constraints," IEEE Trans. Evolutionary Computation,
    vol. 18, no. 4, 2014.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from vamos.engine.algorithm.components.hooks import (
    finalize_genealogy,
    live_should_stop,
    track_offspring_genealogy,
)
from vamos.engine.algorithm.components.termination import HVTracker
from vamos.engine.algorithm.components.utils import variation_operator_label
from vamos.foundation.kernel import default_kernel

from .helpers import (
    evaluate_population_with_constraints,
    nsgaiii_survival,
)
from .initialization import initialize_nsgaiii_run
from .state import NSGAIIIState, build_nsgaiii_result

if TYPE_CHECKING:
    from vamos.foundation.eval.backends import EvaluationBackend
    from vamos.foundation.kernel.backend import KernelBackend
    from vamos.foundation.problem.types import ProblemProtocol
from vamos.foundation.observer import RunContext
from vamos.hooks.live_viz import LiveVisualization

__all__ = ["NSGAIII"]


class NSGAIII:
    """Non-dominated Sorting Genetic Algorithm III for many-objective optimization.

    NSGA-III uses reference points for diversity maintenance, making it suitable
    for problems with 3 or more objectives where crowding distance is less effective.
    Reference points guide the search toward a well-distributed Pareto front.

    Parameters
    ----------
    config : dict
        Algorithm configuration with keys:
        - pop_size (int): Population size (recommended to align with reference points)
        - crossover (tuple): Crossover operator config
        - mutation (tuple): Mutation operator config
        - reference_directions (dict, optional): Reference point configuration
        - selection (tuple, optional): Parent selection config
        - external_archive_size (int, optional): Size of external archive
        - archive_type (str, optional): Archive type ("hypervolume" or "crowding")
        - hv_threshold (float, optional): HV threshold for early termination
    kernel : KernelBackend
        Backend for vectorized operations.

    Examples
    --------
    Basic usage:

    >>> from vamos.algorithms import NSGAIIIConfig
    >>> config = NSGA3Config().pop_size(92).divisions(12).build()
    >>> nsga3 = NSGAIII(config, kernel)
    >>> result = nsga3.run(problem, ("max_evaluations", 20000), seed=42)

    Ask/tell interface:

    >>> nsga3 = NSGAIII(config, kernel)
    >>> nsga3.initialize(problem, ("max_evaluations", 20000), seed=42)
    >>> while not nsga3.should_terminate():
    ...     X = nsga3.ask()
    ...     F = evaluate(X)
    ...     nsga3.tell(F)
    >>> result = nsga3.result()
    """

    def __init__(self, config: dict[str, Any], kernel: KernelBackend | None = None):
        self.cfg = config
        self.kernel = kernel or default_kernel()
        self._st: NSGAIIIState | None = None
        self._live_cb: LiveVisualization | None = None
        self._eval_strategy: EvaluationBackend | None = None
        self._max_eval: int = 0
        self._hv_tracker: HVTracker | None = None
        self._problem: ProblemProtocol | None = None

    # -------------------------------------------------------------------------
    # Main run method (batch mode)
    # -------------------------------------------------------------------------

    def run(
        self,
        problem: ProblemProtocol,
        termination: tuple[str, Any] = ("max_evaluations", 25000),
        seed: int = 0,
        eval_strategy: EvaluationBackend | None = None,
        live_viz: LiveVisualization | None = None,
    ) -> dict[str, Any]:
        """Run NSGA-III optimization loop.

        Parameters
        ----------
        problem : ProblemProtocol
            Problem to optimize.
        termination : tuple
            Termination criterion, e.g., ("max_evaluations", 10000).
        seed : int
            Random seed for reproducibility.
        eval_strategy : EvaluationBackend, optional
            Evaluation backend for parallel evaluation.
        live_viz : LiveVisualization, optional
            Live visualization callback.

        Returns
        -------
        dict
            Result dictionary with X, F, G, reference_directions, archive data.
        """
        self._st, live_cb, eval_strategy, max_eval, hv_tracker = initialize_nsgaiii_run(
            self.cfg, self.kernel, problem, termination, seed, eval_strategy, live_viz
        )
        self._live_cb = live_cb
        self._eval_strategy = eval_strategy
        self._max_eval = max_eval
        self._hv_tracker = cast(HVTracker, hv_tracker)
        self._problem = problem

        st = self._st
        ctx = RunContext(
            problem=problem,
            algorithm=self,
            config=self.cfg,
            algorithm_name="nsgaiii",
            engine_name=str(self.kernel.name),
        )
        live_cb.on_start(ctx)

        hv_reached = False
        stop_requested = False
        while st.n_eval < max_eval and not stop_requested:
            # Generate and evaluate offspring
            X_off = self._generate_offspring(st)

            # Evaluate offspring
            F_off, G_off = self._evaluate_offspring(problem, X_off, eval_strategy, st.constraint_mode)
            st.n_eval += X_off.shape[0]

            # Combine ids for genealogy if tracking
            ids_combined = None
            if st.ids is not None and st.pending_offspring_ids is not None:
                ids_combined = np.concatenate([st.ids, st.pending_offspring_ids])

            # NSGA-III survival selection
            (
                st.X,
                st.F,
                st.G,
                survivor_indices,
                st.ideal_point,
                st.extreme_points,
                st.worst_point,
            ) = nsgaiii_survival(
                st.X,
                st.F,
                st.G,
                X_off,
                F_off,
                G_off,
                st.pop_size,
                st.ref_dirs_norm,
                st.rng,
                st.ideal_point,
                st.extreme_points,
                st.worst_point,
            )

            # Update ids based on survival selection
            if ids_combined is not None:
                st.ids = ids_combined[survivor_indices]

            st.generation += 1

            # Update archive
            if st.archive_manager is not None:
                st.archive_X, st.archive_F = st.archive_manager.update(st.X, st.F, st.G)

            # Live callback
            live_cb.on_generation(st.generation, F=st.F, stats={"evals": st.n_eval})
            stop_requested = live_should_stop(live_cb)

            # Check HV threshold
            if hv_tracker is not None and hv_tracker.enabled and hv_tracker.reached(st.hv_points()):
                hv_reached = True
                break

        live_cb.on_end(final_F=st.F)
        result = build_nsgaiii_result(st, hv_reached, kernel=self.kernel)
        finalize_genealogy(result, st, self.kernel)
        return result

    # -------------------------------------------------------------------------
    # Generation logic
    # -------------------------------------------------------------------------

    def _generate_offspring(self, st: NSGAIIIState) -> np.ndarray:
        """Generate offspring using tournament selection and variation."""
        n_var = st.X.shape[1]
        n_parents = 2 * (st.pop_size // 2)

        ranks, crowd = self.kernel.nsga2_ranking(st.F)
        parents_idx = self.kernel.tournament_selection(ranks, crowd, st.pressure, st.rng, n_parents=n_parents)

        X_parents = st.X[parents_idx].reshape(-1, 2, n_var)
        if st.crossover_fn is None or st.mutation_fn is None:
            raise RuntimeError("NSGA-III variation operators are not initialized.")
        offspring_pairs = st.crossover_fn(X_parents)
        X_off = offspring_pairs.reshape(-1, n_var)
        X_off = st.mutation_fn(X_off)

        # Track genealogy
        if st.genealogy_tracker is not None:
            parent_pairs = parents_idx.flatten()
            track_offspring_genealogy(
                st,
                parent_pairs,
                X_off.shape[0],
                variation_operator_label(self.cfg, "sbx+pm"),
                "nsgaiii",
            )

        return X_off

    def _evaluate_offspring(
        self,
        problem: ProblemProtocol,
        X: np.ndarray,
        eval_strategy: EvaluationBackend,
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
        problem: ProblemProtocol,
        termination: tuple[str, Any] = ("max_evaluations", 25000),
        seed: int = 0,
        eval_strategy: EvaluationBackend | None = None,
        live_viz: LiveVisualization | None = None,
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
        self._st, self._live_cb, self._eval_strategy, self._max_eval, self._hv_tracker = initialize_nsgaiii_run(
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
                algorithm_name="nsgaiii",
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

    def tell(self, eval_result: Any, problem: ProblemProtocol | None = None) -> bool:
        """Receive evaluated offspring and update population.

        Parameters
        ----------
        eval_result : Any
            Objective values as ``np.ndarray``, or an object with ``.F``
            attribute, or a dict with ``"F"`` key.
        problem : ProblemProtocol | None
            Unused, kept for interface consistency.

        Returns
        -------
        bool
            Always ``False`` (NSGA-III has no early-stop criterion via tell).

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
        X_off = st.pending_offspring
        assert X_off is not None
        st.pending_offspring = None

        if hasattr(eval_result, "F"):
            F = np.asarray(eval_result.F, dtype=float)
            G = getattr(eval_result, "G", None)
        elif isinstance(eval_result, dict):
            F = np.asarray(eval_result["F"], dtype=float)
            G = eval_result.get("G")
        else:
            F = np.asarray(eval_result, dtype=float)
            G = None

        # Combine ids for genealogy if tracking
        ids_combined = None
        if st.ids is not None and st.pending_offspring_ids is not None:
            ids_combined = np.concatenate([st.ids, st.pending_offspring_ids])

        # NSGA-III survival selection
        (
            st.X,
            st.F,
            st.G,
            survivor_indices,
            st.ideal_point,
            st.extreme_points,
            st.worst_point,
        ) = nsgaiii_survival(
            st.X,
            st.F,
            st.G,
            X_off,
            F,
            G,
            st.pop_size,
            st.ref_dirs_norm,
            st.rng,
            st.ideal_point,
            st.extreme_points,
            st.worst_point,
        )

        # Update ids based on survival selection
        if ids_combined is not None:
            st.ids = ids_combined[survivor_indices]

        st.n_eval += X_off.shape[0]
        st.generation += 1

        # Update archive
        if st.archive_manager is not None:
            st.archive_X, st.archive_F = st.archive_manager.update(st.X, st.F, st.G)

        # Live callback
        if self._live_cb is not None:
            self._live_cb.on_generation(st.generation, F=st.F)

        # Check HV tracker
        if st.hv_tracker is not None and st.hv_tracker.enabled:
            st.hv_tracker.reached(st.hv_points())

        return False

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
            Result dictionary with X, F, G, reference_directions, archive data.

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

        result = build_nsgaiii_result(self._st, hv_reached, kernel=self.kernel)
        finalize_genealogy(result, self._st, self.kernel)
        return result

    @property
    def state(self) -> NSGAIIIState | None:
        """Access current algorithm state."""
        return self._st
