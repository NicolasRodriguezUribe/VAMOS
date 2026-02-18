"""SMPSO core algorithm implementation.

Speed-constrained Multiobjective Particle Swarm Optimization uses velocity
clamping and crowding-distance leader selection.

Reference:
    Nebro, A.J., Durillo, J.J., Garcia-Nieto, J., Coello Coello, C.A.,
    Luna, F. and Alba, E. (2009). SMPSO: A new PSO-based metaheuristic
    for multi-objective optimization. IEEE MCDM'09, pp. 66-73.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from vamos.engine.algorithm.components.hooks import (
    finalize_genealogy,
    live_should_stop,
    notify_generation,
    track_offspring_genealogy,
)
from vamos.engine.algorithm.components.archive import _single_front_crowding
from .helpers import (
    extract_eval_arrays,
    update_personal_bests,
)
from .initialization import initialize_smpso_run
from .state import SMPSOState, build_smpso_result
from vamos.engine.algorithm.components.termination import HVTracker

if TYPE_CHECKING:
    from vamos.foundation.eval.backends import EvaluationBackend
    from vamos.foundation.kernel.backend import KernelBackend
    from vamos.foundation.problem.types import ProblemProtocol
from vamos.hooks.live_viz import LiveVisualization
from vamos.foundation.observer import RunContext


__all__ = ["SMPSO"]


def _select_global_best(
    arch_X: np.ndarray,
    arch_F: np.ndarray,
    rng: np.random.Generator,
    n_select: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select leaders using crowding-distance binary tournaments.

    Returns the selected leaders and their indices in the archive.
    """
    size = arch_X.shape[0]
    if size == 0:
        raise ValueError("Archive is empty.")
    if size <= 2:
        leaders = np.repeat(arch_X[:1], n_select, axis=0)
        leader_idx = np.zeros(n_select, dtype=int)
        return leaders, leader_idx

    crowding = _single_front_crowding(arch_F)
    leaders = np.empty((n_select, arch_X.shape[1]), dtype=arch_X.dtype)
    leader_idx = np.empty(n_select, dtype=int)
    for i in range(n_select):
        a, b = rng.choice(size, size=2, replace=False)
        ca = crowding[a]
        cb = crowding[b]
        if ca > cb:
            winner = a
        elif cb > ca:
            winner = b
        else:
            winner = a if rng.random() < 0.5 else b
        leaders[i] = arch_X[winner]
        leader_idx[i] = winner
    return leaders, leader_idx


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

    >>> from vamos.algorithms import SMPSOConfig
    >>> config = SMPSOConfig.builder().pop_size(100).archive_size(100).build()
    >>> smpso = SMPSO(config, kernel)
    >>> result = smpso.run(problem, ("max_evaluations", 10000), seed=42)

    Ask/tell interface:

    >>> smpso = SMPSO(config, kernel)
    >>> smpso.initialize(problem, ("max_evaluations", 10000), seed=42)
    >>> while not smpso.should_terminate():
    ...     X = smpso.ask()
    ...     result = evaluate(X)
    ...     smpso.tell(result, problem)
    >>> result = smpso.result()
    """

    def __init__(self, config: dict[str, Any], kernel: KernelBackend):
        self.cfg = config
        self.kernel = kernel
        self._st: SMPSOState | None = None
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
        termination: tuple[str, Any],
        seed: int = 0,
        eval_strategy: EvaluationBackend | None = None,
        live_viz: LiveVisualization | None = None,
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
        eval_strategy : EvaluationBackend, optional
            Evaluation backend.
        live_viz : LiveVisualization, optional
            Live visualization callback.

        Returns
        -------
        dict
            Result dictionary with X, F, archive data.
        """
        self._st, live_cb, eval_strategy, max_eval, hv_tracker = initialize_smpso_run(
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
            algorithm_name="smpso",
            engine_name=str(self.kernel.name),
        )
        live_cb.on_start(ctx)
        live_cb.on_generation(0, F=st.hv_points(), stats={"evals": st.n_eval})
        stop_requested = live_should_stop(live_cb)

        hv_reached = hv_tracker.enabled and hv_tracker.reached(st.hv_points())

        while st.n_eval < max_eval and not hv_reached and not stop_requested:
            X_off = self.ask()
            eval_result = eval_strategy.evaluate(X_off, problem)
            hv_reached = self.tell(eval_result, problem)

            if hv_tracker.enabled and hv_tracker.reached(st.hv_points()):
                hv_reached = True
                break

            stop_requested = notify_generation(live_cb, self.kernel, st.generation, st.hv_points(), stats={"evals": st.n_eval})

        result = build_smpso_result(st, hv_reached)
        finalize_genealogy(result, st, self.kernel)
        live_cb.on_end(final_F=result.get("F"))
        return result

    # -------------------------------------------------------------------------
    # Ask/Tell Interface
    # -------------------------------------------------------------------------

    def initialize(
        self,
        problem: ProblemProtocol,
        termination: tuple[str, Any],
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
        self._st, self._live_cb, self._eval_strategy, self._max_eval, self._hv_tracker = initialize_smpso_run(
            self.cfg, self.kernel, problem, termination, seed, eval_strategy, live_viz
        )
        self._problem = problem
        if self._st is not None and self._live_cb is not None:
            ctx = RunContext(
                problem=problem,
                algorithm=self,
                config=self.cfg,
                algorithm_name="smpso",
                engine_name=str(self.kernel.name),
            )
            self._live_cb.on_start(ctx)
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
            raise RuntimeError("SMPSO is not initialized. Call run() or initialize() first.")
        st = self._st
        if st.pending_offspring is not None:
            raise RuntimeError("Previous offspring not yet consumed by tell().")

        # Select leaders from archive using crowding-based binary tournament
        arch_X = st.archive_X
        arch_F = st.archive_F
        leaders: np.ndarray
        leader_idx: np.ndarray
        if arch_X is None or arch_F is None or arch_F.size == 0:
            leader_idx = np.asarray(st.rng.integers(0, st.X.shape[0], size=st.X.shape[0]), dtype=int)
            leaders = st.X[leader_idx]
        else:
            leaders, leader_idx = _select_global_best(arch_X, arch_F, st.rng, st.X.shape[0])

        # PSO velocity update (SMPSO with constriction)
        pop_size, n_var = st.X.shape
        r1 = np.round(st.rng.uniform(st.r1_min, st.r1_max, size=(pop_size, 1)), 1)
        r2 = np.round(st.rng.uniform(st.r2_min, st.r2_max, size=(pop_size, 1)), 1)
        c1 = np.round(st.rng.uniform(st.c1_min, st.c1_max, size=(pop_size, 1)), 1)
        c2 = np.round(st.rng.uniform(st.c2_min, st.c2_max, size=(pop_size, 1)), 1)
        rho = c1 + c2
        disc = np.maximum(rho * rho - 4.0 * rho, 0.0)
        constriction = np.where(
            rho <= 4.0,
            1.0,
            2.0 / (2.0 - rho - np.sqrt(disc)),
        )
        velocity = constriction * (st.max_weight * st.velocity + c1 * r1 * (st.pbest_X - st.X) + c2 * r2 * (leaders - st.X))
        velocity = np.clip(velocity, st.delta_min, st.delta_max)

        # Position update
        X_new = st.X + velocity
        low_mask = X_new < st.xl
        if np.any(low_mask):
            X_new = np.where(low_mask, st.xl, X_new)
            velocity[low_mask] *= st.change_velocity1
        high_mask = X_new > st.xu
        if np.any(high_mask):
            X_new = np.where(high_mask, st.xu, X_new)
            velocity[high_mask] *= st.change_velocity2

        # Apply mutation (turbulence)
        if st.mutation_op is not None:
            if st.mutation_every <= 1:
                X_new = st.mutation_op(X_new, st.rng)
            else:
                mask = (np.arange(pop_size) % st.mutation_every) == 0
                if np.any(mask):
                    X_new[mask] = st.mutation_op(X_new[mask], st.rng)

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

        return cast(np.ndarray, X_new)

    def tell(
        self,
        eval_result: Any,
        problem: ProblemProtocol | None = None,
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
        assert X_new is not None
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
        hv_reached = self._st.hv_tracker is not None and self._st.hv_tracker.enabled and self._st.hv_tracker.reached(self._st.hv_points())
        result = build_smpso_result(self._st, hv_reached)
        finalize_genealogy(result, self._st, self.kernel)
        return result

    @property
    def state(self) -> SMPSOState | None:
        """Access current algorithm state."""
        return self._st
