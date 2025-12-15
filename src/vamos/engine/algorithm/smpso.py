"""SMPSO: Speed-constrained Multiobjective Particle Swarm Optimization.

This module implements SMPSO, a particle swarm optimizer adapted for
multi-objective problems. It uses velocity clamping to prevent particle
explosion and maintains an external archive of leaders for guidance.

Key features:
    - Vectorized ask/tell loop (batch evaluation friendly)
    - Crowding-distance leader selection from an external archive
    - Polynomial mutation ("turbulence") for exploration
    - Optional HV-based termination (("hv", {...})) via KernelBackend when available
    - Live visualization callbacks
    - Feasibility-aware personal best update for constrained problems

Reference:
    Nebro, A.J., Durillo, J.J., Garcia-Nieto, J., Coello Coello, C.A.,
    Luna, F. and Alba, E. (2009). SMPSO: A new PSO-based metaheuristic
    for multi-objective optimization. IEEE MCDM'09, pp. 66-73.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.engine.algorithm.components.archive import CrowdingDistanceArchive
from vamos.engine.algorithm.components.base import (
    get_eval_backend,
    get_live_viz,
    notify_generation,
    parse_termination,
    setup_hv_tracker,
)
from vamos.engine.algorithm.components.population import initialize_population, resolve_bounds
from vamos.engine.algorithm.components.variation import prepare_mutation_params
from vamos.engine.algorithm.smpso_state import SMPSOState, build_smpso_result
from vamos.engine.operators.real import (
    ClampRepair,
    PolynomialMutation,
    ReflectRepair,
    ResampleRepair,
    RoundRepair,
    VariationWorkspace,
)
from vamos.foundation.constraints.utils import compute_violation, is_feasible

if TYPE_CHECKING:
    from vamos.foundation.eval.backends import EvaluationBackend
    from vamos.foundation.kernel.backend import KernelBackend
    from vamos.foundation.problem.types import ProblemProtocol
    from vamos.ux.visualization.live_viz import LiveVisualization


_REPAIR_MAP = {
    "clip": ClampRepair,
    "clamp": ClampRepair,
    "reflect": ReflectRepair,
    "random": ResampleRepair,
    "resample": ResampleRepair,
    "round": RoundRepair,
}


def _resolve_repair(cfg: Any | None) -> Any | None:
    if cfg is None:
        return None
    if isinstance(cfg, tuple):
        method, params = cfg
        params = dict(params)
    elif isinstance(cfg, dict):
        cfg = dict(cfg)
        method = cfg.pop("method", cfg.pop("name", cfg.pop("type", None)))
        params = cfg
    else:
        method, params = str(cfg), {}

    if method is None:
        return None
    normalized = str(method).lower()
    if normalized in {"none", "off", "disabled"}:
        return None
    cls = _REPAIR_MAP.get(normalized)
    if cls is None:
        raise ValueError(f"Unknown repair strategy '{method}' for SMPSO.")
    return cls(**params)


def _dominates(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    leq = a <= b
    lt = a < b
    return np.all(leq, axis=1) & np.any(lt, axis=1)


def _update_personal_bests(
    X: np.ndarray,
    F: np.ndarray,
    G: np.ndarray | None,
    pbest_X: np.ndarray,
    pbest_F: np.ndarray,
    pbest_G: np.ndarray | None,
    constraint_mode: str,
) -> None:
    if constraint_mode and constraint_mode != "none" and G is not None and pbest_G is not None:
        feas_new = is_feasible(G)
        feas_old = is_feasible(pbest_G)
        cv_new = compute_violation(G)
        cv_old = compute_violation(pbest_G)

        better = np.zeros(X.shape[0], dtype=bool)
        better |= feas_new & ~feas_old
        better |= feas_new & feas_old & _dominates(F, pbest_F)
        better |= (~feas_new) & (~feas_old) & (cv_new < cv_old)
        update_idx = better
    else:
        update_idx = _dominates(F, pbest_F) | np.allclose(F, pbest_F)

    pbest_X[update_idx] = X[update_idx]
    pbest_F[update_idx] = F[update_idx]
    if pbest_G is not None and G is not None:
        pbest_G[update_idx] = G[update_idx]


def _extract_eval_arrays(eval_result: Any) -> tuple[np.ndarray, np.ndarray | None]:
    if hasattr(eval_result, "F"):
        F = eval_result.F
        G = getattr(eval_result, "G", None)
        return F, G
    if isinstance(eval_result, dict):
        return eval_result.get("F"), eval_result.get("G")
    return np.asarray(eval_result, dtype=float), None


class SMPSO:
    """
    Speed-constrained Multiobjective Particle Swarm Optimization.

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
    """

    def __init__(self, config: dict, kernel: "KernelBackend"):
        self.cfg = config
        self.kernel = kernel
        self._st: SMPSOState | None = None
        self._live_cb: LiveVisualization | None = None
        self._eval_backend: EvaluationBackend | None = None
        self._max_eval: int = 0
        self._hv_tracker: Any = None
        self._problem: ProblemProtocol | None = None

    def run(
        self,
        problem: "ProblemProtocol",
        termination: tuple[str, Any],
        seed: int,
        eval_backend: "EvaluationBackend | None" = None,
        live_viz: "LiveVisualization | None" = None,
    ) -> dict[str, Any]:
        live_cb, eval_backend, max_eval, hv_tracker = self._initialize_run(
            problem, termination, seed, eval_backend, live_viz
        )
        st = self._st
        assert st is not None, "State not initialized"

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
        live_cb.on_end(final_F=result.get("F"))
        return result

    def _initialize_run(
        self,
        problem: "ProblemProtocol",
        termination: tuple[str, Any],
        seed: int,
        eval_backend: "EvaluationBackend | None" = None,
        live_viz: "LiveVisualization | None" = None,
    ) -> tuple[Any, Any, int, Any]:
        max_eval, hv_config = parse_termination(termination, "SMPSO")
        eval_backend = get_eval_backend(eval_backend)
        live_cb = get_live_viz(live_viz)
        rng = np.random.default_rng(seed)

        pop_size = int(self.cfg.get("pop_size", 100))
        archive_size = int(self.cfg.get("archive_size", pop_size))
        inertia = float(self.cfg.get("inertia", 0.5))
        c1 = float(self.cfg.get("c1", 1.5))
        c2 = float(self.cfg.get("c2", 1.5))
        vmax_fraction = float(self.cfg.get("vmax_fraction", 0.5))

        encoding = getattr(problem, "encoding", "continuous")
        if encoding not in {"continuous", "real"}:
            raise ValueError("SMPSO currently supports continuous/real encoding only.")

        n_var = int(problem.n_var)
        n_obj = int(problem.n_obj)
        xl, xu = resolve_bounds(problem, encoding)

        span = xu - xl
        vmax = np.abs(span) * vmax_fraction
        vmax[vmax == 0.0] = 1.0

        initializer_cfg = self.cfg.get("initializer")
        X = initialize_population(pop_size, n_var, xl, xu, encoding, rng, problem, initializer=initializer_cfg)

        eval_init = eval_backend.evaluate(X, problem)
        F = eval_init.F
        constraint_mode = self.cfg.get("constraint_mode", "feasibility")
        G = eval_init.G if constraint_mode != "none" else None

        velocity = rng.uniform(-vmax, vmax, size=X.shape)

        pbest_X = X.copy()
        pbest_F = F.copy()
        pbest_G = G.copy() if G is not None else None

        leader_archive = CrowdingDistanceArchive(archive_size, n_var, n_obj, X.dtype)
        archive_X, archive_F = leader_archive.update(X, F)

        mut_method, mut_params = self.cfg.get("mutation", ("pm", {}))
        mut_method = str(mut_method).lower()
        if mut_method not in {"pm", "polynomial"}:
            raise ValueError(f"Unsupported SMPSO mutation '{mut_method}'.")
        mut_params = prepare_mutation_params(dict(mut_params or {}), encoding, n_var)
        workspace = VariationWorkspace()
        mutation_op = PolynomialMutation(
            prob_mutation=float(mut_params.get("prob", 1.0 / max(1, n_var))),
            eta=float(mut_params.get("eta", 20.0)),
            lower=xl,
            upper=xu,
            workspace=workspace,
        )

        repair_op = _resolve_repair(self.cfg.get("repair"))

        hv_tracker = setup_hv_tracker(hv_config, self.kernel)

        live_cb.on_start(problem=problem, algorithm=self, config=self.cfg)

        self._st = SMPSOState(
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
        )

        return live_cb, eval_backend, max_eval, hv_tracker

    def initialize(
        self,
        problem: "ProblemProtocol",
        termination: tuple[str, Any],
        seed: int,
        eval_backend: "EvaluationBackend | None" = None,
        live_viz: "LiveVisualization | None" = None,
    ) -> None:
        self._live_cb, self._eval_backend, self._max_eval, self._hv_tracker = (
            self._initialize_run(problem, termination, seed, eval_backend, live_viz)
        )
        self._problem = problem
        if self._st is not None:
            self._live_cb.on_generation(0, F=self._st.hv_points())

    def ask(self) -> np.ndarray:
        if self._st is None:
            raise RuntimeError("SMPSO is not initialized. Call run() or initialize() first.")
        st = self._st
        if st.pending_offspring is not None:
            raise RuntimeError("Previous offspring not yet consumed by tell().")

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

        r1 = st.rng.random(size=st.X.shape)
        r2 = st.rng.random(size=st.X.shape)
        cognitive = st.c1 * r1 * (st.pbest_X - st.X)
        social = st.c2 * r2 * (leaders - st.X)
        velocity = st.inertia * st.velocity + cognitive + social
        velocity = np.clip(velocity, -st.vmax, st.vmax)

        X_new = st.X + velocity
        np.clip(X_new, st.xl, st.xu, out=X_new)

        if st.mutation_op is not None:
            X_new = st.mutation_op(X_new, st.rng)
        if st.repair_op is not None:
            X_new = st.repair_op(X_new, st.xl, st.xu, st.rng)
        np.clip(X_new, st.xl, st.xu, out=X_new)

        st.velocity = velocity
        st.pending_offspring = X_new
        return X_new

    def tell(self, eval_result: Any, problem: "ProblemProtocol | None" = None) -> bool:
        if self._st is None or self._st.pending_offspring is None:
            raise RuntimeError("Call ask() before tell().")
        st = self._st

        X_new = st.pending_offspring
        st.pending_offspring = None

        F, G = _extract_eval_arrays(eval_result)
        if st.constraint_mode == "none":
            G = None

        st.n_eval += X_new.shape[0]
        st.generation += 1

        _update_personal_bests(
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

        if st.archive_manager is not None:
            st.archive_X, st.archive_F = st.archive_manager.update(X_new, F)

        if st.hv_tracker is not None and st.hv_tracker.enabled:
            return st.hv_tracker.reached(st.hv_points())
        return False

    def should_terminate(self) -> bool:
        if self._st is None:
            return True
        if self._st.n_eval >= self._max_eval:
            return True
        if self._hv_tracker is not None and self._hv_tracker.enabled:
            return self._hv_tracker.reached(self._st.hv_points())
        return False

    def result(self) -> dict[str, Any]:
        if self._st is None:
            raise RuntimeError("SMPSO is not initialized.")
        hv_reached = (
            self._st.hv_tracker is not None
            and self._st.hv_tracker.enabled
            and self._st.hv_tracker.reached(self._st.hv_points())
        )
        return build_smpso_result(self._st, hv_reached)

    @property
    def state(self) -> SMPSOState | None:
        return self._st


__all__ = ["SMPSO"]

