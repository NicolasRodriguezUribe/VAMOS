from __future__ import annotations

import numpy as np

from vamos.algorithm.archive import CrowdingDistanceArchive, _single_front_crowding
from vamos.algorithm.population import (
    evaluate_population,
    evaluate_population_with_constraints,
    initialize_population,
    resolve_bounds,
)
from vamos.constraints.utils import compute_violation, is_feasible
from vamos.algorithm.variation import prepare_mutation_params
from vamos.operators.real import PolynomialMutation, VariationWorkspace, ClampRepair, ReflectRepair, ResampleRepair, RoundRepair


_REPAIR_MAP = {
    "clip": ClampRepair,
    "clamp": ClampRepair,
    "reflect": ReflectRepair,
    "random": ResampleRepair,
    "resample": ResampleRepair,
    "round": RoundRepair,
}


def _resolve_repair(cfg):
    if cfg is None:
        return None
    if isinstance(cfg, tuple):
        method, params = cfg
        params = dict(params)
    elif isinstance(cfg, dict):
        method = cfg.get("method") or cfg.get("name") or cfg.get("type")
        params = {k: v for k, v in cfg.items() if k not in {"method", "name", "type"}}
    else:
        method = str(cfg)
        params = {}
    if method is None:
        return None
    method = method.lower()
    if method == "none":
        return None
    cls = _REPAIR_MAP.get(method)
    if cls is None:
        raise ValueError(f"Unknown repair strategy '{method}' for SMPSO.")
    return cls(**params)


def _dominates(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    leq = a <= b
    lt = a < b
    return np.all(leq, axis=1) & np.any(lt, axis=1)


def _select_leaders(F_archive: np.ndarray, rng: np.random.Generator, count: int) -> np.ndarray:
    if F_archive.shape[0] == 0:
        return np.empty((0,), dtype=int)
    crowd = _single_front_crowding(F_archive)
    if not np.isfinite(crowd).any():
        idx = rng.integers(0, F_archive.shape[0], size=count)
    else:
        probs = crowd / np.sum(crowd[np.isfinite(crowd)]) if np.sum(crowd[np.isfinite(crowd)]) > 0 else None
        if probs is None or not np.isfinite(probs).any():
            idx = rng.integers(0, F_archive.shape[0], size=count)
        else:
            probs = probs / probs.sum()
            idx = rng.choice(F_archive.shape[0], size=count, p=probs)
    return idx


class SMPSO:
    """
    Speed-constrained Multiobjective PSO with an external leaders archive.
    """

    def __init__(self, config: dict, kernel):
        self.cfg = config
        self.kernel = kernel

    def run(self, problem, termination, seed: int):
        term_type, term_val = termination
        assert term_type == "n_eval", "Only termination=('n_eval', N) is supported."
        max_eval = int(term_val)

        rng = np.random.default_rng(seed)
        pop_size = int(self.cfg["pop_size"])
        archive_size = int(self.cfg.get("archive_size", pop_size))
        encoding = getattr(problem, "encoding", "continuous")
        if encoding not in {"continuous", "real"}:
            raise ValueError("SMPSO currently supports continuous/real encoding only.")
        n_var = problem.n_var
        xl, xu = resolve_bounds(problem, encoding)
        span = xu - xl
        vmax = np.abs(span) * float(self.cfg.get("vmax_fraction", 0.5))
        vmax[vmax == 0.0] = 1.0

        initializer_cfg = self.cfg.get("initializer")
        X = initialize_population(pop_size, n_var, xl, xu, encoding, rng, problem, initializer=initializer_cfg)
        velocity = rng.uniform(-vmax, vmax, size=X.shape)

        constraint_mode = self.cfg.get("constraint_mode", "none")
        if constraint_mode and constraint_mode != "none":
            F, G = evaluate_population_with_constraints(problem, X)
        else:
            F = evaluate_population(problem, X)
            G = None
        n_eval = X.shape[0]

        pbest_X = X.copy()
        pbest_F = F.copy()
        pbest_G = G.copy() if G is not None else None

        archive_manager = CrowdingDistanceArchive(archive_size, n_var, problem.n_obj, X.dtype)
        arch_X, arch_F = archive_manager.update(X, F)
        arch_G = G.copy() if G is not None else None

        inertia = float(self.cfg.get("inertia", 0.5))
        c1 = float(self.cfg.get("c1", 1.5))
        c2 = float(self.cfg.get("c2", 1.5))

        mut_method, mut_params = self.cfg["mutation"]
        mut_params = prepare_mutation_params(mut_params, encoding, n_var, prob_factor=self.cfg.get("mutation_prob_factor"))
        workspace = VariationWorkspace()
        mutation = PolynomialMutation(
            prob_mutation=float(mut_params.get("prob", 1.0 / max(1, n_var))),
            eta=float(mut_params.get("eta", 20.0)),
            lower=xl,
            upper=xu,
            workspace=workspace,
        )
        repair_op = _resolve_repair(self.cfg.get("repair"))

        while n_eval < max_eval:
            if arch_F.shape[0] == 0:
                leader_idx = rng.integers(0, X.shape[0], size=X.shape[0])
                leaders = X[leader_idx]
            else:
                leader_idx = _select_leaders(arch_F, rng, X.shape[0])
                leaders = arch_X[leader_idx]

            r1 = rng.random(size=X.shape)
            r2 = rng.random(size=X.shape)
            cognitive = c1 * r1 * (pbest_X - X)
            social = c2 * r2 * (leaders - X)
            velocity = inertia * velocity + cognitive + social
            velocity = np.clip(velocity, -vmax, vmax)

            X = X + velocity
            X = np.clip(X, xl, xu)
            X = mutation(X, rng)
            if repair_op is not None:
                X = repair_op(X, xl, xu, rng)

            if constraint_mode and constraint_mode != "none":
                F, G = evaluate_population_with_constraints(problem, X)
            else:
                F = evaluate_population(problem, X)
                G = None
            n_eval += X.shape[0]

            self._update_personal_bests(X, F, G, pbest_X, pbest_F, pbest_G, constraint_mode)

            if G is not None:
                arch_G = G.copy()
            arch_X, arch_F = archive_manager.update(X, F)

        result_F = arch_F if arch_F.size else F
        result_X = arch_X if arch_X.size else X
        result = {
            "X": result_X,
            "F": result_F,
            "evaluations": n_eval,
            "archive": {"X": arch_X, "F": arch_F},
        }
        if constraint_mode != "none" and G is not None:
            result["G"] = G
        return result

    def _update_personal_bests(self, X, F, G, pbest_X, pbest_F, pbest_G, constraint_mode: str):
        if constraint_mode and constraint_mode != "none" and G is not None:
            feas_new = is_feasible(G)
            feas_old = is_feasible(pbest_G) if pbest_G is not None else np.ones_like(feas_new, dtype=bool)
            cv_new = compute_violation(G)
            cv_old = compute_violation(pbest_G) if pbest_G is not None else np.zeros_like(cv_new)

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
