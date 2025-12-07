"""
Support functions for the NSGA-II evolutionary loop.
Separated to keep nsgaii.py focused on the class orchestration.
"""
from __future__ import annotations

import numpy as np

from vamos.constraints.utils import compute_violation, is_feasible
from vamos.analytics.genealogy import GenealogyTracker, get_lineage


def build_mating_pool(
    kernel,
    ranks: np.ndarray,
    crowding: np.ndarray,
    pressure: int,
    rng: np.random.Generator,
    parent_count: int,
    group_size: int = 2,
    selection_method: str = "tournament",
) -> np.ndarray:
    if parent_count <= 0:
        raise ValueError("parent_count must be positive.")
    if parent_count % group_size != 0:
        raise ValueError("parent_count must be divisible by group_size.")
    assert ranks.shape == crowding.shape, "ranks and crowding must align"
    if selection_method == "random":
        parent_indices = rng.integers(0, ranks.size, size=parent_count)
    else:
        parent_indices = kernel.tournament_selection(
            ranks, crowding, pressure, rng, n_parents=parent_count
        )
    if parent_indices.size != parent_count:
        raise ValueError("Selection operator returned an unexpected number of parents.")
    return parent_indices.reshape(parent_count // group_size, group_size)


def feasible_nsga2_survival(kernel, X, F, G, X_off, F_off, G_off, pop_size):
    """
    Feasibility rule:
      - Feasible dominate infeasible.
      - Among feasible: standard NSGA-II rank/crowding.
      - Among infeasible: lower sum violation wins.
    """
    X_comb = np.vstack([X, X_off])
    F_comb = np.vstack([F, F_off])
    G_comb = np.vstack([G, G_off]) if G is not None and G_off is not None else None
    if G_comb is None:
        X_sel, F_sel = kernel.nsga2_survival(X, F, X_off, F_off, pop_size)
        return X_sel, F_sel, None

    feas = is_feasible(G_comb)
    cv = compute_violation(G_comb)
    selected = []

    if feas.any():
        feas_idx = np.nonzero(feas)[0]
        ranks, crowd = kernel.nsga2_ranking(F_comb[feas_idx])
        # Order feasible by rank then crowding
        order = np.lexsort((-crowd, ranks))
        feas_ordered = feas_idx[order]
        selected.extend(feas_ordered.tolist())

    if len(selected) < pop_size:
        infeas_idx = np.nonzero(~feas)[0]
        if infeas_idx.size:
            order_infeas = infeas_idx[np.argsort(cv[infeas_idx])]
            selected.extend(order_infeas.tolist())

    selected = selected[:pop_size]
    return X_comb[selected], F_comb[selected], G_comb[selected]


def match_ids(new_X: np.ndarray, combined_X: np.ndarray, combined_ids: np.ndarray) -> np.ndarray:
    """
    Map surviving rows in new_X back to their ids from combined_X/combined_ids.
    Uses an isclose match; falls back to -1 when no match is found.
    """
    new_ids = np.full(new_X.shape[0], -1, dtype=int)
    for i, row in enumerate(new_X):
        matches = np.where(np.all(np.isclose(combined_X, row, atol=1e-8), axis=1))[0]
        if matches.size:
            new_ids[i] = combined_ids[matches[0]]
    return new_ids


def operator_success_stats(tracker: GenealogyTracker, final_ids: list[int]) -> list[dict]:
    final_ancestors = set()
    for fid in final_ids:
        for rec in get_lineage(tracker, fid):
            final_ancestors.add(rec.individual_id)
    totals = {}
    finals = {}
    for rec in tracker.records.values():
        if rec.operator_name is None:
            continue
        op = rec.operator_name
        totals[op] = totals.get(op, 0) + 1
        if rec.individual_id in final_ancestors:
            finals[op] = finals.get(op, 0) + 1
    rows = []
    for op, cnt in totals.items():
        good = finals.get(op, 0)
        rows.append(
            {
                "operator": op,
                "total_uses": cnt,
                "uses_in_final_lineages": good,
                "ratio": (good / cnt) if cnt else 0.0,
            }
        )
    return rows


def generation_contributions(tracker: GenealogyTracker, final_ids: list[int]) -> list[dict]:
    final_ancestors = set()
    for fid in final_ids:
        for rec in get_lineage(tracker, fid):
            final_ancestors.add(rec.individual_id)
    gen_totals = {}
    gen_final = {}
    for rec in tracker.records.values():
        gen_totals[rec.generation] = gen_totals.get(rec.generation, 0) + 1
        if rec.individual_id in final_ancestors:
            gen_final[rec.generation] = gen_final.get(rec.generation, 0) + 1
    rows = []
    for gen in sorted(gen_totals.keys()):
        tot = gen_totals.get(gen, 0)
        fin = gen_final.get(gen, 0)
        rows.append(
            {
                "generation": gen,
                "total": tot,
                "final_lineage": fin,
                "ratio": (fin / tot) if tot else 0.0,
            }
        )
    return rows


__all__ = [
    "build_mating_pool",
    "feasible_nsga2_survival",
    "match_ids",
    "operator_success_stats",
    "generation_contributions",
]
