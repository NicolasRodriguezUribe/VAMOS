from __future__ import annotations

import numpy as np

from vamos.engine.algorithm.nsgaii import _build_mating_pool
from vamos.engine.algorithm.population import (
    evaluate_population,
    evaluate_population_with_constraints,
    initialize_population,
    resolve_bounds,
)
from vamos.engine.algorithm.variation import VariationPipeline, prepare_mutation_params
from vamos.engine.algorithm.hypervolume import hypervolume
from vamos.foundation.constraints.utils import compute_violation, is_feasible
from vamos.engine.operators.real import VariationWorkspace


def _epsilon_indicator(F: np.ndarray) -> np.ndarray:
    diff = F[:, None, :] - F[None, :, :]
    return np.max(diff, axis=2)


def _hypervolume_indicator(F: np.ndarray) -> np.ndarray:
    n = F.shape[0]
    if n == 0:
        return np.empty((0, 0))
    ref = np.max(F, axis=0) + 1.0
    indicator = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pair = np.vstack([F[i], F[j]])
            hv_pair = hypervolume(pair, ref)
            hv_j = hypervolume(F[j : j + 1], ref)
            indicator[i, j] = hv_j - hv_pair
    return indicator


def _compute_indicator_matrix(F: np.ndarray, indicator: str) -> np.ndarray:
    if indicator == "hypervolume":
        return _hypervolume_indicator(F)
    return _epsilon_indicator(F)


def _ibea_fitness(indicator: np.ndarray, kappa: float) -> np.ndarray:
    mat = indicator.copy()
    np.fill_diagonal(mat, np.inf)
    contrib = np.exp(-mat / kappa)
    contrib[~np.isfinite(contrib)] = 0.0
    return -np.sum(contrib, axis=0)


def _apply_constraint_penalty(fitness: np.ndarray, G: np.ndarray | None) -> np.ndarray:
    if G is None:
        return fitness
    cv = compute_violation(G)
    feas = is_feasible(G)
    if not feas.any():
        return fitness + cv
    penalty = np.max(np.abs(fitness)) + 1.0
    penalized = fitness.copy()
    penalized[~feas] += penalty * (1.0 + cv[~feas])
    return penalized


def _environmental_selection(
    X: np.ndarray,
    F: np.ndarray,
    G: np.ndarray | None,
    pop_size: int,
    indicator: str,
    kappa: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    ind = _compute_indicator_matrix(F, indicator)
    fitness = _ibea_fitness(ind, kappa)
    fitness = _apply_constraint_penalty(fitness, G)

    while X.shape[0] > pop_size:
        worst = int(np.argmax(fitness))
        delta = np.exp(-ind[worst] / kappa)
        delta[worst] = 0.0
        fitness -= delta
        X = np.delete(X, worst, axis=0)
        F = np.delete(F, worst, axis=0)
        if G is not None:
            G = np.delete(G, worst, axis=0)
        ind = np.delete(np.delete(ind, worst, axis=0), worst, axis=1)
        fitness = np.delete(fitness, worst, axis=0)
    return X, F, G, fitness


class IBEA:
    """
    Indicator-Based Evolutionary Algorithm with additive epsilon or hypervolume indicator.
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
        offspring_size = pop_size
        encoding = getattr(problem, "encoding", "continuous")
        n_var = problem.n_var
        xl, xu = resolve_bounds(problem, encoding)

        initializer_cfg = self.cfg.get("initializer")
        X = initialize_population(pop_size, n_var, xl, xu, encoding, rng, problem, initializer=initializer_cfg)
        constraint_mode = self.cfg.get("constraint_mode", "none")
        if constraint_mode and constraint_mode != "none":
            F, G = evaluate_population_with_constraints(problem, X)
        else:
            F = evaluate_population(problem, X)
            G = None
        n_eval = X.shape[0]

        sel_method, sel_params = self.cfg["selection"]
        pressure = int(sel_params.get("pressure", 2))

        cross_method, cross_params = self.cfg["crossover"]
        cross_method = cross_method.lower()
        cross_params = dict(cross_params)

        mut_method, mut_params = self.cfg["mutation"]
        mut_method = mut_method.lower()
        mut_params = prepare_mutation_params(mut_params, encoding, n_var, prob_factor=self.cfg.get("mutation_prob_factor"))

        variation_workspace = VariationWorkspace()
        variation = VariationPipeline(
            encoding=encoding,
            cross_method=cross_method,
            cross_params=cross_params,
            mut_method=mut_method,
            mut_params=mut_params,
            xl=xl,
            xu=xu,
            workspace=variation_workspace,
            repair_cfg=self.cfg.get("repair"),
            problem=problem,
        )

        indicator = self.cfg.get("indicator", "eps").lower()
        kappa = float(self.cfg.get("kappa", 0.05))

        while n_eval < max_eval:
            _, _, _, fitness = _environmental_selection(X.copy(), F.copy(), G.copy() if G is not None else None, pop_size, indicator, kappa)
            ranks = np.argsort(np.argsort(fitness))
            crowd = np.zeros_like(fitness, dtype=float)
            parents_per_group = variation.parents_per_group
            children_per_group = variation.children_per_group
            parent_count = int(np.ceil(offspring_size / children_per_group) * parents_per_group)
            mating_pairs = _build_mating_pool(
                self.kernel, ranks, crowd, pressure, rng, parent_count, parents_per_group, sel_method
            )
            parent_idx = mating_pairs.reshape(-1)
            X_parents = variation.gather_parents(X, parent_idx)
            X_off = variation.produce_offspring(X_parents, rng)
            if X_off.shape[0] > offspring_size:
                X_off = X_off[:offspring_size]

            if constraint_mode and constraint_mode != "none":
                F_off, G_off = evaluate_population_with_constraints(problem, X_off)
            else:
                F_off = evaluate_population(problem, X_off)
                G_off = None
            n_eval += X_off.shape[0]

            X_comb = np.vstack([X, X_off])
            F_comb = np.vstack([F, F_off])
            if G is not None or G_off is not None:
                if G is None:
                    G_comb = G_off
                elif G_off is None:
                    G_comb = G
                else:
                    G_comb = np.vstack([G, G_off])
            else:
                G_comb = None

            X, F, G, _ = _environmental_selection(
                X_comb, F_comb, G_comb, pop_size, indicator, kappa
            )

        result = {"X": X, "F": F, "evaluations": n_eval}
        if G is not None and constraint_mode != "none":
            result["G"] = G
        return result
