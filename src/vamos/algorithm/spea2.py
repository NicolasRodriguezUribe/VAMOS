from __future__ import annotations

import math
import numpy as np

from vamos.algorithm.nsgaii import _build_mating_pool
from vamos.algorithm.population import (
    evaluate_population,
    evaluate_population_with_constraints,
    initialize_population,
    resolve_bounds,
)
from vamos.algorithm.variation import VariationPipeline, prepare_mutation_params
from vamos.constraints.utils import compute_violation, is_feasible
from vamos.operators.real import VariationWorkspace


def _dominance_matrix(F: np.ndarray, G: np.ndarray | None, constraint_mode: str) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Compute dominance matrix with optional feasibility-aware handling.
    dom[i, j] = True if i dominates j.
    """
    N = F.shape[0]
    if N == 0:
        return np.zeros((0, 0), dtype=bool), None, None
    less_equal = F[:, None, :] <= F[None, :, :]
    strictly_less = F[:, None, :] < F[None, :, :]
    dom_obj = np.logical_and(np.all(less_equal, axis=2), np.any(strictly_less, axis=2))
    if constraint_mode == "none" or G is None:
        np.fill_diagonal(dom_obj, False)
        return dom_obj, None, None

    feas = is_feasible(G)
    cv = compute_violation(G)
    dom = np.zeros_like(dom_obj, dtype=bool)
    feas_mat = feas[:, None]
    infeas_mat = ~feas_mat

    dom |= dom_obj & feas_mat & feas_mat.T  # feasible vs feasible by objectives
    dom |= feas_mat & infeas_mat.T  # feasible always dominates infeasible
    # Infeasible vs infeasible: smaller violation wins
    dom |= infeas_mat & infeas_mat.T & (cv[:, None] < cv[None, :])
    np.fill_diagonal(dom, False)
    return dom, feas, cv


def _spea2_fitness(F: np.ndarray, dom_matrix: np.ndarray, k_neighbors: int | None) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute SPEA2 fitness components: raw fitness (dominance-based) and density.
    """
    N = F.shape[0]
    strength = dom_matrix.sum(axis=1)
    raw = dom_matrix.T @ strength

    if N <= 1:
        density = np.zeros(N, dtype=float)
        return raw + density, np.zeros((N, N), dtype=float)
    dist = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    k = k_neighbors if k_neighbors is not None else int(math.sqrt(N))
    k = max(1, min(k, N - 1)) if N > 1 else 1
    kth = np.partition(dist, k, axis=1)[:, k]
    density = 1.0 / (kth + 2.0)
    fitness = raw + density
    return fitness, dist


def _truncate_by_distance(dist_matrix: np.ndarray, keep: int) -> np.ndarray:
    """
    SPEA2 truncation operator: iteratively remove the solution with smallest
    nearest-neighbor distance until only `keep` remain.
    """
    candidates = list(range(dist_matrix.shape[0]))
    if dist_matrix.shape[0] <= keep:
        return np.asarray(candidates, dtype=int)
    dist = dist_matrix.copy()
    while len(candidates) > keep:
        sub = dist[np.ix_(candidates, candidates)]
        np.fill_diagonal(sub, np.inf)
        nearest = np.partition(sub, 1, axis=1)[:, 1]
        remove_pos = int(np.argmin(nearest))
        del candidates[remove_pos]
    return np.asarray(candidates, dtype=int)


def _environmental_selection(
    X: np.ndarray,
    F: np.ndarray,
    G: np.ndarray | None,
    archive_size: int,
    k_neighbors: int | None,
    constraint_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    dom, _, _ = _dominance_matrix(F, G, constraint_mode)
    fitness, dist = _spea2_fitness(F, dom, k_neighbors)

    selected = np.flatnonzero(fitness < 1.0)
    if selected.size == 0:
        order = np.argsort(fitness)
        selected = order[: min(archive_size, F.shape[0])]

    if selected.size > archive_size:
        rel = _truncate_by_distance(dist[np.ix_(selected, selected)], archive_size)
        selected = selected[rel]
    elif selected.size < archive_size:
        remaining = np.setdiff1d(np.arange(F.shape[0]), selected, assume_unique=True)
        if remaining.size:
            order = remaining[np.argsort(fitness[remaining])]
            needed = archive_size - selected.size
            selected = np.concatenate([selected, order[:needed]])

    return X[selected], F[selected], G[selected] if G is not None else None


class SPEA2:
    """
    Strength Pareto Evolutionary Algorithm 2 with external archive.
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

        archive_X, archive_F, archive_G = _environmental_selection(
            X, F, G, archive_size, self.cfg.get("k_neighbors"), constraint_mode
        )

        while n_eval < max_eval:
            ranks, crowding = self._selection_metrics(archive_F, archive_G, constraint_mode)
            parents_per_group = variation.parents_per_group
            children_per_group = variation.children_per_group
            parent_count = int(np.ceil(offspring_size / children_per_group) * parents_per_group)
            mating_pairs = _build_mating_pool(
                self.kernel, ranks, crowding, pressure, rng, parent_count, parents_per_group, sel_method
            )
            parent_idx = mating_pairs.reshape(-1)
            X_parents = variation.gather_parents(archive_X, parent_idx)
            X_off = variation.produce_offspring(X_parents, rng)
            if X_off.shape[0] > offspring_size:
                X_off = X_off[:offspring_size]

            if constraint_mode and constraint_mode != "none":
                F_off, G_off = evaluate_population_with_constraints(problem, X_off)
            else:
                F_off = evaluate_population(problem, X_off)
                G_off = None
            n_eval += X_off.shape[0]

            # Update population and archive
            X = X_off
            F = F_off
            G = G_off
            X_union = np.vstack([X, archive_X])
            F_union = np.vstack([F, archive_F])
            G_union = None
            if constraint_mode != "none" and (G is not None or archive_G is not None):
                if G is None:
                    G_union = archive_G
                elif archive_G is None:
                    G_union = G
                else:
                    G_union = np.vstack([G, archive_G])
            archive_X, archive_F, archive_G = _environmental_selection(
                X_union, F_union, G_union, archive_size, self.cfg.get("k_neighbors"), constraint_mode
            )

        result = {"X": archive_X, "F": archive_F, "evaluations": n_eval}
        if archive_G is not None and constraint_mode != "none":
            result["G"] = archive_G
        result["archive"] = {"X": archive_X, "F": archive_F}
        result["population"] = {"X": X, "F": F}
        return result

    def _selection_metrics(self, F: np.ndarray, G: np.ndarray | None, constraint_mode: str):
        if F.size == 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=float)
        if constraint_mode and constraint_mode != "none" and G is not None:
            cv = compute_violation(G)
            feas = is_feasible(G)
            if feas.any():
                feas_idx = np.nonzero(feas)[0]
                ranks, crowd = self.kernel.nsga2_ranking(F[feas_idx])
                metrics_rank = np.full(F.shape[0], ranks.max(initial=0) + 1, dtype=int)
                metrics_crowd = np.zeros(F.shape[0], dtype=float)
                metrics_rank[feas_idx] = ranks
                metrics_crowd[feas_idx] = crowd
                metrics_crowd[~feas] = -cv[~feas]
                return metrics_rank, metrics_crowd
            ranks = np.zeros(F.shape[0], dtype=int)
            crowd = -cv
            return ranks, crowd
        return self.kernel.nsga2_ranking(F)
