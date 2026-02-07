"""
Ask/tell operations for NSGA-II.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING, cast

import numpy as np

from vamos.foundation.metrics.hypervolume import hypervolume
from vamos.foundation.metrics.pareto import pareto_filter

from .helpers import (
    build_mating_pool,
    compute_crowding,
    fronts_from_ranks,
    incremental_insert_fronts,
    select_nsga2,
)
from .state import compute_selection_metrics, track_offspring_genealogy, update_archives

if TYPE_CHECKING:
    from .nsgaii import NSGAII
    from .state import NSGAIIState


def combine_ids(st: NSGAIIState) -> np.ndarray | None:
    if not st.track_genealogy:
        return None
    current_ids = st.ids if st.ids is not None else np.array([], dtype=int)
    pending_ids = st.pending_offspring_ids if st.pending_offspring_ids is not None else np.array([], dtype=int)
    return cast(np.ndarray, np.concatenate([current_ids, pending_ids]))


def _coerce_parent_candidates(raw: Any, size: int) -> np.ndarray | None:
    if raw is None:
        return None
    arr = np.asarray(raw)
    if arr.ndim != 1:
        return None
    if arr.dtype == bool:
        if arr.size != size:
            return None
        return np.flatnonzero(arr)
    try:
        idx = arr.astype(int, copy=False)
    except Exception:
        return None
    idx = idx[(idx >= 0) & (idx < size)]
    if idx.size == 0:
        return None
    return np.unique(idx)


def ask_nsgaii(algo: NSGAII) -> np.ndarray:
    st = algo._st
    if st is None:
        raise RuntimeError("ask() called before initialization.")

    if st.aos_controller is not None:
        aos_step = st.step
        st.aos_controller.start_generation(aos_step)
        arm = st.aos_controller.select_arm(mating_id=0, batch_size=st.offspring_size)
        idx = st.aos_controller.portfolio.index_of(arm.op_id)
        st.variation = st.operator_pool[idx]
        st.aos_last_op_id = arm.op_id
        st.aos_last_op_name = arm.name
        st.aos_last_batch_size = st.offspring_size
        st.aos_step = aos_step

    if st.incremental_enabled and st.ranks is not None and st.crowding is not None and st.G is None and st.constraint_mode == "none":
        ranks, crowding = st.ranks, st.crowding
    else:
        ranks, crowding = compute_selection_metrics(algo.kernel, st.F, st.G, st.constraint_mode)
        if st.steady_state:
            st.ranks = ranks
            st.crowding = crowding
            st.fronts = fronts_from_ranks(ranks)
    parents_per_group = st.variation.parents_per_group
    children_per_group = st.variation.children_per_group
    parent_count = int(np.ceil(st.offspring_size / children_per_group) * parents_per_group)

    candidate_indices = np.arange(st.X.shape[0], dtype=int)
    filter_fn = st.parent_selection_filter
    if callable(filter_fn):
        selected_raw: Any | None = None
        try:
            selected_raw = filter_fn(st, ranks, crowding)
        except TypeError:
            try:
                selected_raw = filter_fn(st)
            except Exception:
                selected_raw = None
        except Exception:
            selected_raw = None
        selected_idx = _coerce_parent_candidates(selected_raw, st.X.shape[0])
        if selected_idx is not None and selected_idx.size > 0:
            candidate_indices = selected_idx

    if st.non_breeding_indices.size > 0:
        blocked = np.asarray(st.non_breeding_indices, dtype=int)
        blocked = blocked[(blocked >= 0) & (blocked < st.X.shape[0])]
        if blocked.size > 0:
            candidate_indices = candidate_indices[~np.isin(candidate_indices, blocked)]
            if candidate_indices.size == 0:
                candidate_indices = np.arange(st.X.shape[0], dtype=int)

    mating_pairs = build_mating_pool(
        algo.kernel,
        ranks,
        crowding,
        st.pressure,
        st.rng,
        parent_count,
        parents_per_group,
        st.sel_method,
        candidate_indices=candidate_indices,
    )
    parent_idx = mating_pairs.reshape(-1)
    if st.immigration_manager is not None:
        st.immigration_manager.record_parent_indices(st.generation, parent_idx)
    X_parents = st.variation.gather_parents(st.X, parent_idx)
    X_off = st.variation.produce_offspring(X_parents, st.rng)

    if X_off.shape[0] > st.offspring_size:
        X_off = X_off[: st.offspring_size]
    st.pending_offspring = X_off

    if st.aos_controller is not None and st.aos_last_op_id is not None:
        st.aos_last_batch_size = X_off.shape[0]
        st.aos_controller.observe_offspring(st.aos_last_op_id, X_off.shape[0])

    track_offspring_genealogy(st, parent_idx, X_off.shape[0])
    return X_off


def tell_nsgaii(algo: NSGAII, eval_result: Any, pop_size: int) -> bool:
    st = algo._st
    if st is None:
        raise RuntimeError("tell() called before initialization.")

    X_off = st.pending_offspring
    st.pending_offspring = None
    if X_off is None:
        raise ValueError("tell() called without a pending ask().")

    F_off = eval_result.F
    G_off = eval_result.G if st.constraint_mode != "none" else None
    aos_controller = st.aos_controller
    assert st.hv_tracker is not None

    combined_X = np.vstack([st.X, X_off])
    combined_F = np.vstack([st.F, F_off])
    combined_ids = combine_ids(st)
    parent_count = st.X.shape[0]
    selected_idx = None
    prev_F = st.F
    used_incremental = False

    early_reject = False
    if st.steady_state and st.fronts is not None and st.constraint_mode == "none" and st.G is None and G_off is None and F_off is not None:
        worst_front = st.fronts[-1] if st.fronts else []
        if worst_front:
            F_worst = st.F[np.asarray(worst_front, dtype=int)]
            F_off_arr = np.asarray(F_off, dtype=float)
            if F_off_arr.ndim == 1:
                F_off_arr = F_off_arr.reshape(1, -1)
            less_equal = F_worst[:, None, :] <= F_off_arr[None, :, :]
            strictly_less = F_worst[:, None, :] < F_off_arr[None, :, :]
            dominates = np.all(less_equal, axis=2) & np.any(strictly_less, axis=2)
            dominated_by_worst = np.any(dominates, axis=0)
            if dominated_by_worst.size and bool(np.all(dominated_by_worst)):
                early_reject = True

    use_incremental = (
        st.incremental_enabled
        and st.replacement_size == 1
        and st.constraint_mode == "none"
        and st.G is None
        and G_off is None
        and X_off.shape[0] == 1
    )
    if early_reject:
        new_X = st.X
        new_F = st.F
        new_G = st.G
        if aos_controller is not None:
            selected_idx = np.arange(parent_count, dtype=int)
        used_incremental = True
    elif use_incremental:
        if st.fronts is None or st.ranks is None or st.crowding is None:
            ranks, crowding = algo.kernel.nsga2_ranking(st.F)
            st.ranks = ranks
            st.crowding = crowding
            st.fronts = fronts_from_ranks(ranks)

        fronts = [list(front) for front in (st.fronts or [])]
        ranks = np.concatenate([st.ranks or np.empty(0, dtype=int), np.array([-1], dtype=int)])
        incremental_insert_fronts(fronts, ranks, combined_F, combined_F.shape[0] - 1)
        crowding = compute_crowding(combined_F, fronts)

        selected_idx = select_nsga2(fronts, crowding, pop_size)
        new_X = combined_X[selected_idx]
        new_F = combined_F[selected_idx]
        new_G = None

        new_ranks = ranks[selected_idx]
        new_fronts = fronts_from_ranks(new_ranks)
        new_crowding = compute_crowding(new_F, new_fronts)

        st.fronts = new_fronts
        st.ranks = new_ranks
        st.crowding = new_crowding
        used_incremental = True
    elif st.G is None or G_off is None or st.constraint_mode == "none":
        if aos_controller is not None:
            new_X, new_F, selected_idx = algo.kernel.nsga2_survival(st.X, st.F, X_off, F_off, pop_size, return_indices=True)
        else:
            new_X, new_F = algo.kernel.nsga2_survival(st.X, st.F, X_off, F_off, pop_size)
        new_G = None
    else:
        from .helpers import feasible_nsga2_survival

        if aos_controller is not None:
            new_X, new_F, new_G, selected_idx = feasible_nsga2_survival(
                algo.kernel, st.X, st.F, st.G, X_off, F_off, G_off, pop_size, return_indices=True
            )
        else:
            new_X, new_F, new_G = feasible_nsga2_survival(algo.kernel, st.X, st.F, st.G, X_off, F_off, G_off, pop_size)

    if combined_ids is not None:
        from .helpers import match_ids

        st.ids = match_ids(new_X, combined_X, combined_ids)

    st.X, st.F, st.G = new_X, new_F, new_G
    st.pending_offspring_ids = None

    if st.incremental_enabled and not used_incremental:
        ranks, crowding = algo.kernel.nsga2_ranking(st.F)
        st.ranks = ranks
        st.crowding = crowding
        st.fronts = fronts_from_ranks(ranks)

    if aos_controller is not None and selected_idx is not None and st.aos_last_op_id is not None:

        def _normalized_hv(F: np.ndarray | None, ref: np.ndarray, hv_ref: float) -> float:
            if F is None:
                return 0.0
            front = pareto_filter(np.asarray(F, dtype=float))
            if front is None or front.size == 0:
                return 0.0
            if front.ndim != 2 or ref.ndim != 1 or front.shape[1] != ref.shape[0]:
                return 0.0
            front = front[np.all(front <= ref, axis=1)]
            if front.size == 0:
                return 0.0
            hv = float(hypervolume(front, ref, allow_ref_expand=False))
            return hv / hv_ref if hv_ref > 0.0 else 0.0

        hv_delta_rate = 0.0
        try:
            reward_weights = getattr(aos_controller.config, "reward_weights", {}) or {}
            hv_weight = float(reward_weights.get("hv_delta", 0.0))
            reward_scope = str(getattr(aos_controller.config, "reward_scope", "combined") or "combined").lower()
            wants_hv = hv_weight > 0.0 or reward_scope in {"hv", "hv_delta", "hypervolume"}
            if wants_hv:
                hv_delta_rate = 0.5
            hv_ref_point = getattr(aos_controller.config, "hv_reference_point", None)
            hv_ref_hv = getattr(aos_controller.config, "hv_reference_hv", None)
            if wants_hv and hv_ref_point is not None and hv_ref_hv is not None:
                ref = np.asarray(hv_ref_point, dtype=float)
                hv_ref = float(hv_ref_hv)
                hv_prev = _normalized_hv(prev_F, ref, hv_ref)
                hv_new = _normalized_hv(new_F, ref, hv_ref)
                denom = abs(hv_prev) if abs(hv_prev) > 1e-12 else 1e-12
                ratio = (hv_new - hv_prev) / denom
                hv_delta_rate = float(0.5 + 0.5 * np.tanh(ratio))
            elif (
                wants_hv
                and prev_F is not None
                and new_F is not None
                and prev_F.ndim == 2
                and new_F.ndim == 2
                and prev_F.shape[1] == new_F.shape[1]
                and prev_F.shape[1] <= 3
                and prev_F.size > 0
                and new_F.size > 0
            ):
                ref = np.maximum(np.max(prev_F, axis=0), np.max(new_F, axis=0)) + 1.0
                hv_prev = float(algo.kernel.hypervolume(prev_F, ref))
                hv_new = float(algo.kernel.hypervolume(new_F, ref))
                denom = abs(hv_prev) if abs(hv_prev) > 1e-12 else 1e-12
                ratio = (hv_new - hv_prev) / denom
                hv_delta_rate = float(0.5 + 0.5 * np.tanh(ratio))
        except Exception:
            hv_delta_rate = 0.0

        try:
            ranks, _ = algo.kernel.nsga2_ranking(st.F)
            nd_mask = ranks == ranks.min(initial=0)
        except (ValueError, IndexError):
            nd_mask = np.zeros(st.F.shape[0], dtype=bool)
        is_offspring = selected_idx >= parent_count
        n_survivors = int(np.sum(is_offspring))
        n_nd_insertions = int(np.sum(is_offspring & nd_mask))
        aos_controller.observe_survivors(st.aos_last_op_id, n_survivors)
        aos_controller.observe_nd_insertions(st.aos_last_op_id, n_nd_insertions)
        trace_rows = aos_controller.finalize_generation(st.aos_step or 0, hv_delta_rate=hv_delta_rate)
        for row in trace_rows:
            st.aos_trace_rows.append(
                {
                    "step": row.step,
                    "mating_id": row.mating_id,
                    "op_id": row.op_id,
                    "op_name": row.op_name,
                    "reward": row.reward,
                    "reward_survival": row.reward_survival,
                    "reward_nd_insertions": row.reward_nd_insertions,
                    "reward_hv_delta": row.reward_hv_delta,
                    "batch_size": row.batch_size,
                }
            )

    if early_reject:
        update_archives(st, algo.kernel, X=st.X, F=st.F)
    else:
        update_archives(st, algo.kernel, X=combined_X, F=combined_F)

    hv_reached = st.hv_tracker.enabled and st.hv_tracker.reached(st.hv_points_fn())

    return hv_reached


__all__ = ["ask_nsgaii", "tell_nsgaii", "combine_ids"]
