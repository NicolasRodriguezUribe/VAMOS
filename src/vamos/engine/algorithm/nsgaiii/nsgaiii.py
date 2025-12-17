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

from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.engine.algorithm.components.base import (
    finalize_genealogy,
    track_offspring_genealogy,
)
from .helpers import (
    evaluate_population_with_constraints,
    nsgaiii_survival,
)
from .initialization import initialize_nsgaiii_run
from .state import NSGAIIIState, build_nsgaiii_result

if TYPE_CHECKING:
    from vamos.engine.algorithm.components.base import EvaluationBackend, LiveVisualization
    from vamos.foundation.kernel.protocols import KernelBackend
    from vamos.foundation.problem.protocol import ProblemProtocol


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
        - pop_size (int): Population size (should align with reference points)
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

    >>> from vamos import NSGA3Config
    >>> config = NSGA3Config().pop_size(92).divisions(12).fixed()
    >>> nsga3 = NSGAIII(config, kernel)
    >>> result = nsga3.run(problem, ("n_eval", 20000), seed=42)

    Ask/tell interface:

    >>> nsga3 = NSGAIII(config, kernel)
    >>> nsga3.initialize(problem, ("n_eval", 20000), seed=42)
    >>> while not nsga3.should_terminate():
    ...     X = nsga3.ask()
    ...     F = evaluate(X)
    ...     nsga3.tell(X, F)
    >>> result = nsga3.result()
    """

    def __init__(self, config: dict, kernel: "KernelBackend"):
        self.cfg = config
        self.kernel = kernel
        self._st: NSGAIIIState | None = None
        self._live_cb: "LiveVisualization | None" = None
        self._eval_backend: "EvaluationBackend | None" = None
        self._max_eval: int = 0
        self._hv_tracker: Any = None
        self._problem: "ProblemProtocol | None" = None

    # -------------------------------------------------------------------------
    # Main run method (batch mode)
    # -------------------------------------------------------------------------

    def run(
        self,
        problem: "ProblemProtocol",
        termination: tuple[str, Any],
        seed: int,
        eval_backend: "EvaluationBackend | None" = None,
        live_viz: "LiveVisualization | None" = None,
    ) -> dict[str, Any]:
        """Run NSGA-III optimization loop.

        Parameters
        ----------
        problem : ProblemProtocol
            Problem to optimize.
        termination : tuple
            Termination criterion, e.g., ("n_eval", 10000).
        seed : int
            Random seed for reproducibility.
        eval_backend : EvaluationBackend, optional
            Evaluation backend for parallel evaluation.
        live_viz : LiveVisualization, optional
            Live visualization callback.

        Returns
        -------
        dict
            Result dictionary with X, F, G, reference_directions, archive data.
        """
        self._st, live_cb, eval_backend, max_eval, hv_tracker = initialize_nsgaiii_run(
            self.cfg, self.kernel, problem, termination, seed, eval_backend, live_viz
        )
        self._live_cb = live_cb
        self._eval_backend = eval_backend
        self._max_eval = max_eval
        self._hv_tracker = hv_tracker
        self._problem = problem

        st = self._st
        live_cb.on_start(problem=problem, algorithm=self, config=self.cfg)

        hv_reached = False
        while st.n_eval < max_eval:
            # Generate and evaluate offspring
            X_off = self._generate_offspring(st)

            # Evaluate offspring
            F_off, G_off = self._evaluate_offspring(
                problem, X_off, eval_backend, st.constraint_mode
            )
            st.n_eval += X_off.shape[0]

            # Combine ids for genealogy if tracking
            ids_combined = None
            if st.ids is not None and st.pending_offspring_ids is not None:
                ids_combined = np.concatenate([st.ids, st.pending_offspring_ids])

            # NSGA-III survival selection
            st.X, st.F, st.G, survivor_indices = nsgaiii_survival(
                st.X, st.F, st.G, X_off, F_off, G_off, st.pop_size, st.ref_dirs_norm, st.rng
            )

            # Update ids based on survival selection
            if ids_combined is not None:
                st.ids = ids_combined[survivor_indices]

            st.generation += 1

            # Update archive
            if st.archive_manager is not None:
                st.archive_manager.update(st.X, st.F)
                st.archive_X, st.archive_F = st.archive_manager.get_archive()

            # Live callback
            live_cb.on_generation(st.generation, F=st.F)

            # Check HV threshold
            if hv_tracker is not None:
                hv_tracker.update(st.F)
                if hv_tracker.reached_threshold():
                    hv_reached = True
                    break

        live_cb.on_end(final_F=st.F)
        result = build_nsgaiii_result(st, hv_reached)
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
        parents_idx = self.kernel.tournament_selection(
            ranks, crowd, st.pressure, st.rng, n_parents=n_parents
        )

        X_parents = st.X[parents_idx].reshape(-1, 2, n_var)
        offspring_pairs = st.crossover_fn(X_parents)
        X_off = offspring_pairs.reshape(-1, n_var)
        X_off = st.mutation_fn(X_off)

        # Track genealogy
        if st.genealogy_tracker is not None:
            parent_pairs = parents_idx.flatten()
            track_offspring_genealogy(st, parent_pairs, X_off.shape[0], "sbx+pm", "nsgaiii")

        return X_off

    def _evaluate_offspring(
        self,
        problem: "ProblemProtocol",
        X: np.ndarray,
        eval_backend: "EvaluationBackend",
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
        eval_backend: "EvaluationBackend | None" = None,
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
        eval_backend : EvaluationBackend, optional
            Evaluation backend.
        live_viz : LiveVisualization, optional
            Live visualization callback.
        """
        self._st, self._live_cb, self._eval_backend, self._max_eval, self._hv_tracker = (
            initialize_nsgaiii_run(
                self.cfg, self.kernel, problem, termination, seed, eval_backend, live_viz
            )
        )
        self._problem = problem
        if self._st is not None:
            self._st.pending_offspring = None
        if self._live_cb is not None:
            self._live_cb.on_start(problem=problem, algorithm=self, config=self.cfg)

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

        # Combine ids for genealogy if tracking
        ids_combined = None
        if st.ids is not None and st.pending_offspring_ids is not None:
            ids_combined = np.concatenate([st.ids, st.pending_offspring_ids])

        # NSGA-III survival selection
        st.X, st.F, st.G, survivor_indices = nsgaiii_survival(
            st.X, st.F, st.G, X, F, G, st.pop_size, st.ref_dirs_norm, st.rng
        )

        # Update ids based on survival selection
        if ids_combined is not None:
            st.ids = ids_combined[survivor_indices]

        st.n_eval += X.shape[0]
        st.generation += 1

        # Update archive
        if st.archive_manager is not None:
            st.archive_manager.update(st.X, st.F)
            st.archive_X, st.archive_F = st.archive_manager.get_archive()

        # Live callback
        if self._live_cb is not None:
            self._live_cb.on_generation(st.generation, F=st.F)

        # Check HV tracker
        if st.hv_tracker is not None:
            st.hv_tracker.update(st.F)

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
        if self._st.hv_tracker is not None and self._st.hv_tracker.reached_threshold():
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

        hv_reached = (
            self._st.hv_tracker is not None
            and self._st.hv_tracker.reached_threshold()
        )

        if self._live_cb is not None:
            self._live_cb.on_end(final_F=self._st.F)

        result = build_nsgaiii_result(self._st, hv_reached)
        finalize_genealogy(result, self._st, self.kernel)
        return result

    @property
    def state(self) -> NSGAIIIState | None:
        """Access current algorithm state."""
        return self._st
