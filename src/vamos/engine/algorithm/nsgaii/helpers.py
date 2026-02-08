"""
Support functions for the NSGA-II evolutionary loop.
Separated to keep the main algorithm class focused on orchestration.
"""

from __future__ import annotations

from typing import Any, Literal, overload

import numpy as np

from vamos.foundation.constraints.utils import compute_violation, is_feasible
from vamos.hooks.genealogy import GenealogyTracker, get_lineage


def build_mating_pool(
    kernel: Any,
    ranks: np.ndarray,
    crowding: np.ndarray,
    pressure: int,
    rng: np.random.Generator,
    parent_count: int,
    group_size: int = 2,
    selection_method: str = "tournament",
    candidate_indices: np.ndarray | None = None,
) -> np.ndarray:
    if parent_count <= 0:
        raise ValueError("parent_count must be positive.")
    if parent_count % group_size != 0:
        raise ValueError("parent_count must be divisible by group_size.")
    assert ranks.shape == crowding.shape, "ranks and crowding must align"
    if candidate_indices is not None:
        cand = np.asarray(candidate_indices, dtype=int)
        cand = cand[(cand >= 0) & (cand < ranks.size)]
        cand = np.unique(cand)
        if cand.size == 0:
            cand = np.arange(ranks.size, dtype=int)
    else:
        cand = np.arange(ranks.size, dtype=int)

    if selection_method == "random":
        parent_indices = rng.choice(cand, size=parent_count, replace=True)
    else:
        parent_local = kernel.tournament_selection(
            ranks[cand],
            crowding[cand],
            pressure,
            rng,
            n_parents=parent_count,
        )
        parent_indices = cand[np.asarray(parent_local, dtype=int)]
    if parent_indices.size != parent_count:
        raise ValueError("Selection operator returned an unexpected number of parents.")
    return parent_indices.reshape(parent_count // group_size, group_size)


@overload
def feasible_nsga2_survival(
    kernel: Any,
    X: np.ndarray,
    F: np.ndarray,
    G: np.ndarray | None,
    X_off: np.ndarray,
    F_off: np.ndarray,
    G_off: np.ndarray | None,
    pop_size: int,
    return_indices: Literal[False] = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]: ...


@overload
def feasible_nsga2_survival(
    kernel: Any,
    X: np.ndarray,
    F: np.ndarray,
    G: np.ndarray | None,
    X_off: np.ndarray,
    F_off: np.ndarray,
    G_off: np.ndarray | None,
    pop_size: int,
    return_indices: Literal[True],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]: ...


def feasible_nsga2_survival(
    kernel: Any,
    X: np.ndarray,
    F: np.ndarray,
    G: np.ndarray | None,
    X_off: np.ndarray,
    F_off: np.ndarray,
    G_off: np.ndarray | None,
    pop_size: int,
    return_indices: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
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
        if return_indices:
            X_sel, F_sel, sel = kernel.nsga2_survival(X, F, X_off, F_off, pop_size, return_indices=True)
            return X_sel, F_sel, None, sel
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
    if return_indices:
        return X_comb[selected], F_comb[selected], G_comb[selected], np.asarray(selected, dtype=int)
    return X_comb[selected], F_comb[selected], G_comb[selected]


def match_ids(new_X: np.ndarray, combined_X: np.ndarray, combined_ids: np.ndarray) -> np.ndarray:
    """
    Map surviving rows in new_X back to their ids from combined_X/combined_ids.
    Uses an exact row match; falls back to -1 when no match is found.
    """
    new_ids = np.full(new_X.shape[0], -1, dtype=int)
    for i, row in enumerate(new_X):
        matches = np.where(np.all(combined_X == row, axis=1))[0]
        if matches.size:
            new_ids[i] = combined_ids[matches[0]]
    return new_ids


def operator_success_stats(tracker: GenealogyTracker, final_ids: list[int]) -> list[dict[str, object]]:
    final_ancestors = set()
    for fid in final_ids:
        for rec in get_lineage(tracker, fid):
            final_ancestors.add(rec.individual_id)
    totals: dict[str, int] = {}
    finals: dict[str, int] = {}
    for rec in tracker.records.values():
        if rec.operator_name is None:
            continue
        op = rec.operator_name
        totals[op] = totals.get(op, 0) + 1
        if rec.individual_id in final_ancestors:
            finals[op] = finals.get(op, 0) + 1
    rows: list[dict[str, object]] = []
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


def generation_contributions(tracker: GenealogyTracker, final_ids: list[int]) -> list[dict[str, object]]:
    final_ancestors = set()
    for fid in final_ids:
        for rec in get_lineage(tracker, fid):
            final_ancestors.add(rec.individual_id)
    gen_totals: dict[int, int] = {}
    gen_final: dict[int, int] = {}
    for rec in tracker.records.values():
        gen_totals[rec.generation] = gen_totals.get(rec.generation, 0) + 1
        if rec.individual_id in final_ancestors:
            gen_final[rec.generation] = gen_final.get(rec.generation, 0) + 1
    rows: list[dict[str, object]] = []
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


def fronts_from_ranks(ranks: np.ndarray) -> list[list[int]]:
    if ranks.size == 0:
        return []
    max_rank = int(ranks.max(initial=-1))
    return [np.flatnonzero(ranks == r).tolist() for r in range(max_rank + 1)]


def compute_crowding(F: np.ndarray, fronts: list[list[int]]) -> np.ndarray:
    """Compute crowding distance for the provided fronts."""
    N = F.shape[0]
    crowding = np.zeros(N, dtype=float)
    for front in fronts:
        if not front:
            continue
        front_arr = np.asarray(front, dtype=int)
        if front_arr.size == 1:
            crowding[front_arr[0]] = np.inf
            continue
        fvals = F[front_arr]
        n_obj = fvals.shape[1]
        d = np.zeros(front_arr.size, dtype=float)
        for m in range(n_obj):
            order = np.argsort(fvals[:, m], kind="mergesort")
            sorted_vals = fvals[order, m]
            d[order[0]] = np.inf
            d[order[-1]] = np.inf
            span = sorted_vals[-1] - sorted_vals[0]
            if span <= 0.0:
                continue
            contrib = np.zeros_like(sorted_vals)
            contrib[1:-1] = (sorted_vals[2:] - sorted_vals[:-2]) / span
            d[order[1:-1]] += contrib[1:-1]
        crowding[front_arr] = d
    return crowding


def select_nsga2(fronts: list[list[int]], crowding: np.ndarray, pop_size: int) -> np.ndarray:
    """NSGA-II elitist selection based on fronts + crowding."""
    selected: list[int] = []
    for front in fronts:
        if not front:
            continue
        front_arr = np.asarray(front, dtype=int)
        if len(selected) + front_arr.size <= pop_size:
            selected.extend(front_arr.tolist())
        else:
            rem = pop_size - len(selected)
            order = np.argsort(crowding[front_arr])[::-1]
            selected.extend(front_arr[order[:rem]].tolist())
            break
    return np.asarray(selected, dtype=int)


def incremental_insert_fronts(
    fronts: list[list[int]],
    ranks: np.ndarray,
    F: np.ndarray,
    new_idx: int,
) -> list[int]:
    """
    Incrementally insert a single solution into existing non-dominated fronts.

    Assumes fronts are valid and indices are sorted. Returns affected front indices.
    """
    affected: list[int] = []
    f_new = F[new_idx]
    inserted = False

    for i, front in enumerate(fronts):
        if not front:
            continue
        front_arr = np.asarray(front, dtype=int)
        f_front = F[front_arr]
        dominates_new = np.all(f_front <= f_new, axis=1) & np.any(f_front < f_new, axis=1)
        if np.any(dominates_new):
            continue

        dominates_front = np.all(f_new <= f_front, axis=1) & np.any(f_new < f_front, axis=1)
        dominated = front_arr[dominates_front].tolist()

        if dominated:
            front = [idx for idx in front if idx not in dominated]
        front.append(new_idx)
        front.sort()
        fronts[i] = front
        ranks[new_idx] = i
        affected.append(i)

        displaced = dominated
        j = i + 1
        while displaced:
            if j >= len(fronts):
                fronts.append(sorted(displaced))
                for idx in displaced:
                    ranks[idx] = j
                affected.append(j)
                break
            front_j = fronts[j]
            front_j_arr = np.asarray(front_j, dtype=int)
            f_front_j = F[front_j_arr]
            dom_mask = np.zeros(front_j_arr.shape[0], dtype=bool)
            for idx in displaced:
                f_idx = F[idx]
                dom_mask |= np.all(f_idx <= f_front_j, axis=1) & np.any(f_idx < f_front_j, axis=1)
            dominated_j = front_j_arr[dom_mask].tolist()
            if dominated_j:
                front_j = [idx for idx in front_j if idx not in dominated_j]
            front_j.extend(displaced)
            front_j.sort()
            fronts[j] = front_j
            for idx in displaced:
                ranks[idx] = j
            affected.append(j)
            displaced = dominated_j
            j += 1

        inserted = True
        break

    if not inserted:
        fronts.append([new_idx])
        ranks[new_idx] = len(fronts) - 1
        affected.append(len(fronts) - 1)
    return affected


def reindex_fronts(fronts: list[list[int]], removed_idx: int) -> list[list[int]]:
    """Remove index from fronts and shift indices above it down by one."""
    new_fronts: list[list[int]] = []
    for front in fronts:
        updated: list[int] = []
        for idx in front:
            if idx == removed_idx:
                continue
            updated.append(idx - 1 if idx > removed_idx else idx)
        if updated:
            new_fronts.append(updated)
    return new_fronts


def ranks_from_fronts(fronts: list[list[int]], size: int) -> np.ndarray:
    ranks = np.empty(size, dtype=int)
    for rank, front in enumerate(fronts):
        ranks[np.asarray(front, dtype=int)] = rank
    return ranks


__all__ = [
    "build_mating_pool",
    "feasible_nsga2_survival",
    "match_ids",
    "operator_success_stats",
    "generation_contributions",
    "fronts_from_ranks",
    "compute_crowding",
    "select_nsga2",
    "incremental_insert_fronts",
    "reindex_fronts",
    "ranks_from_fronts",
]
