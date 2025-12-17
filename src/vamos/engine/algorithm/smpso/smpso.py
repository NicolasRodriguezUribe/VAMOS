"""SMPSO core algorithm implementation.

Speed-constrained Multiobjective Particle Swarm Optimization uses velocity
clamping and crowding-distance leader selection.

Reference:
    Nebro, A.J., Durillo, J.J., Garcia-Nieto, J., Coello Coello, C.A.,
    Luna, F. and Alba, E. (2009). SMPSO: A new PSO-based metaheuristic
    for multi-objective optimization. IEEE MCDM'09, pp. 66-73.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.engine.algorithm.components.base import (
    finalize_genealogy,
    notify_generation,
    track_offspring_genealogy,
)
from .helpers import (
    extract_eval_arrays,
    update_personal_bests,
)
from .initialization import initialize_smpso_run
from .state import SMPSOState, build_smpso_result

if TYPE_CHECKING:
    from vamos.engine.algorithm.components.base import EvaluationBackend, LiveVisualization
    from vamos.foundation.kernel.protocols import KernelBackend
    from vamos.foundation.problem.protocol import ProblemProtocol


__all__ = ["SMPSO"]


class SMPSO:
    """Speed-constrained Multiobjective Particle Swarm Optimization.

    Parameters
    ----------
    config : dict
        Algorithm configuration with keys:
        - pop_size (int): swarm size
        - archive_size (int): leaders archive capacity
        - mutation (tuple): mutation config, e.g. ("pm", {"prob": "1/n", "eta": 20})
        - inertia, c1, c2, vmax_fraction (float): PSO parameters
        - initializer (dict, optional): initializer config (lhs/scatter/random)
        - repair (tuple, optional): bounds repair strategy
        - constraint_mode (str, optional): "none" or feasibility mode
    kernel : KernelBackend
        Backend for vectorized kernels (ranking, HV, etc.).

    Examples
    --------
    Basic usage:

    >>> from vamos import SMPSOConfig
    >>> config = SMPSOConfig().pop_size(100).archive_size(100).fixed()
    >>> smpso = SMPSO(config, kernel)
    >>> result = smpso.run(problem, ("n_eval", 10000), seed=42)

    Ask/tell interface:

    >>> smpso = SMPSO(config, kernel)
    >>> smpso.initialize(problem, ("n_eval", 10000), seed=42)
    >>> while not smpso.should_terminate():
    ...     X = smpso.ask()
    ...     result = evaluate(X)
    ...     smpso.tell(result, problem)
    >>> result = smpso.result()
    """

    def __init__(self, config: dict, kernel: "KernelBackend"):
        self.cfg = config
        self.kernel = kernel
        self._st: SMPSOState | None = None
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
        """Run SMPSO optimization loop.

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

        Returns
        -------
        dict
            Result dictionary with X, F, archive data.
        """
        self._st, live_cb, eval_backend, max_eval, hv_tracker = initialize_smpso_run(
            self.cfg, self.kernel, problem, termination, seed, eval_backend, live_viz
        )
        self._live_cb = live_cb
        self._eval_backend = eval_backend
        self._max_eval = max_eval
        self._hv_tracker = hv_tracker
        self._problem = problem

        st = self._st
        live_cb.on_start(problem=problem, algorithm=self, config=self.cfg)
        live_cb.on_generation(0, F=st.hv_points())

        hv_reached = hv_tracker.enabled and hv_tracker.reached(st.hv_points())

        while st.n_eval < max_eval and not hv_reached:
            X_off = self.ask()
            eval_result = eval_backend.evaluate(X_off, problem)
            hv_reached = self.tell(eval_result, problem)

            if hv_tracker.enabled and hv_tracker.reached(st.hv_points()):
                hv_reached = True
                break

            notify_generation(live_cb, self.kernel, st.generation, st.hv_points())

        result = build_smpso_result(st, hv_reached)
        finalize_genealogy(result, st, self.kernel)
        live_cb.on_end(final_F=result.get("F"))
        return result

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
            initialize_smpso_run(
                self.cfg, self.kernel, problem, termination, seed, eval_backend, live_viz
            )
        )
        self._problem = problem
        if self._st is not None and self._live_cb is not None:
            self._live_cb.on_start(problem=problem, algorithm=self, config=self.cfg)
            self._live_cb.on_generation(0, F=self._st.hv_points())

    def ask(self) -> np.ndarray:
        """Generate offspring positions for evaluation.

        Updates particle velocities using PSO update rule with cognitive
        and social components, applies mutation for turbulence.

        Returns
        -------
        np.ndarray
            New particle positions to evaluate.

        Raises
        ------
        RuntimeError
            If algorithm not initialized or previous offspring not consumed.
        """
        if self._st is None:
            raise RuntimeError(
                "SMPSO is not initialized. Call run() or initialize() first."
            )
        st = self._st
        if st.pending_offspring is not None:
            raise RuntimeError("Previous offspring not yet consumed by tell().")

        # Select leaders from archive using tournament selection
        arch_X = st.archive_X
        arch_F = st.archive_F
        if arch_X is None or arch_F is None or arch_F.size == 0:
            leader_idx = st.rng.integers(0, st.X.shape[0], size=st.X.shape[0])
            leaders = st.X[leader_idx]
        else:
            ranks, crowding = self.kernel.nsga2_ranking(arch_F)
            leader_idx = self.kernel.tournament_selection(
                ranks=ranks,
                crowding=crowding,
                pressure=2,
                rng=st.rng,
                n_parents=st.X.shape[0],
            )
            leaders = arch_X[leader_idx]

        # PSO velocity update
        r1 = st.rng.random(size=st.X.shape)
        r2 = st.rng.random(size=st.X.shape)
        cognitive = st.c1 * r1 * (st.pbest_X - st.X)
        social = st.c2 * r2 * (leaders - st.X)
        velocity = st.inertia * st.velocity + cognitive + social
        velocity = np.clip(velocity, -st.vmax, st.vmax)

        # Position update
        X_new = st.X + velocity
        np.clip(X_new, st.xl, st.xu, out=X_new)

        # Apply mutation (turbulence)
        if st.mutation_op is not None:
            X_new = st.mutation_op(X_new, st.rng)

        # Apply repair
        if st.repair_op is not None:
            X_new = st.repair_op(X_new, st.xl, st.xu, st.rng)
        np.clip(X_new, st.xl, st.xu, out=X_new)

        st.velocity = velocity
        st.pending_offspring = X_new

        # Track genealogy: for PSO, each particle is child of itself + leader
        if st.genealogy_tracker is not None:
            pop_size = st.X.shape[0]
            self_idx = np.arange(pop_size)
            parent_pairs = np.column_stack([self_idx, leader_idx]).flatten()
            track_offspring_genealogy(st, parent_pairs, pop_size, "pso_update", "smpso")

        return X_new

    def tell(
        self,
        eval_result: Any,
        problem: "ProblemProtocol | None" = None,
    ) -> bool:
        """Receive evaluated offspring and update swarm.

        Parameters
        ----------
        eval_result : Any
            Evaluation result with F and optional G.
        problem : ProblemProtocol, optional
            Problem (for API compatibility).

        Returns
        -------
        bool
            True if HV threshold reached.

        Raises
        ------
        RuntimeError
            If algorithm not initialized or no pending offspring.
        """
        if self._st is None or self._st.pending_offspring is None:
            raise RuntimeError("Call ask() before tell().")
        st = self._st

        X_new = st.pending_offspring
        st.pending_offspring = None

        F, G = extract_eval_arrays(eval_result)
        if st.constraint_mode == "none":
            G = None

        st.n_eval += X_new.shape[0]
        st.generation += 1

        # Update personal bests
        update_personal_bests(
            X_new,
            F,
            G,
            st.pbest_X,
            st.pbest_F,
            st.pbest_G,
            st.constraint_mode,
        )

        st.X = X_new
        st.F = F
        st.G = G

        # Update genealogy ids
        if st.pending_offspring_ids is not None:
            st.ids = st.pending_offspring_ids
            st.pending_offspring_ids = None

        # Update leader archive
        if st.archive_manager is not None:
            st.archive_X, st.archive_F = st.archive_manager.update(X_new, F)

        if st.hv_tracker is not None and st.hv_tracker.enabled:
            return st.hv_tracker.reached(st.hv_points())
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
        if self._hv_tracker is not None and self._hv_tracker.enabled:
            return self._hv_tracker.reached(self._st.hv_points())
        return False

    def result(self) -> dict[str, Any]:
        """Get current result.

        Returns
        -------
        dict
            Result dictionary with X, F, archive data.

        Raises
        ------
        RuntimeError
            If algorithm not initialized.
        """
        if self._st is None:
            raise RuntimeError("SMPSO is not initialized.")
        hv_reached = (
            self._st.hv_tracker is not None
            and self._st.hv_tracker.enabled
            and self._st.hv_tracker.reached(self._st.hv_points())
        )
        result = build_smpso_result(self._st, hv_reached)
        finalize_genealogy(result, self._st, self.kernel)
        return result

    @property
    def state(self) -> SMPSOState | None:
        """Access current algorithm state."""
        return self._st
