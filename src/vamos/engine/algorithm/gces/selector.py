"""
Phase-1 GCES split-front selector.

This module intentionally operates only on the split front. The surrounding
merge, ranking, and front-filling flow stays in the GCES algorithm host.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np

from vamos.foundation.quality_indicators.hypervolume import hypervolume as _hypervolume

_GAP_EPS = 1e-12
_CURVGAP_LAMBDA = 0.7
_HV_REF_POINT_VALUE = 1.1
_REFCOVER_DIVISIONS_2D = 100
_REFCOVER_DIVISIONS_3D = 12
_SECTOR_DIVISIONS_3D = 12

_SELECTOR_NUMBA_DISABLED = False
_PAIRWISE_DISTANCES_JIT = None
_REFERENCE_DISTANCE_MATRIX_JIT = None
_REFERENCE_COVER_GAINS_JIT = None


@dataclass(frozen=True)
class _Component:
    indices: np.ndarray
    mst_edges: tuple[tuple[int, int], ...]
    weight: float


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    kernel = _get_pairwise_distances_jit()
    if kernel is not None:
        return np.asarray(kernel(np.asarray(points, dtype=np.float64)), dtype=float)
    diff = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(diff, axis=2)


def _get_numba_njit() -> Any | None:
    global _SELECTOR_NUMBA_DISABLED  # noqa: PLW0603
    if _SELECTOR_NUMBA_DISABLED:
        return None
    try:
        from numba import njit
    except ImportError:
        _SELECTOR_NUMBA_DISABLED = True
        return None
    return njit


def _get_pairwise_distances_jit() -> Any | None:
    global _PAIRWISE_DISTANCES_JIT  # noqa: PLW0603
    if _PAIRWISE_DISTANCES_JIT is not None:
        return _PAIRWISE_DISTANCES_JIT
    njit = _get_numba_njit()
    if njit is None:
        return None

    @njit(cache=True)
    def _kernel(points: np.ndarray) -> np.ndarray:
        n_points, n_obj = points.shape
        distances = np.empty((n_points, n_points), dtype=np.float64)
        for i in range(n_points):
            distances[i, i] = 0.0
            for j in range(i + 1, n_points):
                total = 0.0
                for obj in range(n_obj):
                    diff = points[i, obj] - points[j, obj]
                    total += diff * diff
                dist = math.sqrt(total)
                distances[i, j] = dist
                distances[j, i] = dist
        return distances

    _PAIRWISE_DISTANCES_JIT = _kernel
    return _PAIRWISE_DISTANCES_JIT


def _normalize_by_bounds(points: np.ndarray, ideal: np.ndarray, nadir: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(points, dtype=float)
    spans = nadir - ideal
    valid = spans > 0.0
    if np.any(valid):
        normalized[:, valid] = (points[:, valid] - ideal[valid]) / spans[valid]
    return normalized


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    scores = np.asarray(values, dtype=float)
    if scores.size == 0:
        return np.empty(0, dtype=float)
    lo = float(np.min(scores))
    hi = float(np.max(scores))
    if hi <= lo:
        return np.zeros_like(scores, dtype=float)
    return (scores - lo) / (hi - lo)


def _require_two_objectives(points: np.ndarray, selector_name: str) -> None:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{selector_name} currently supports only 2-objective split fronts.")


def _sorted_front_order_2d(normalized: np.ndarray) -> np.ndarray:
    _require_two_objectives(normalized, "2-objective gap selectors")
    n_points = int(normalized.shape[0])
    return np.asarray(
        np.lexsort(
            (
                np.arange(n_points, dtype=int),
                normalized[:, 1],
                normalized[:, 0],
            )
        ),
        dtype=int,
    )


def _ideal_distances(normalized: np.ndarray) -> np.ndarray:
    return np.asarray(np.linalg.norm(normalized, axis=1), dtype=float)


def _preserved_extremes_2d(normalized: np.ndarray, ideal_distances: np.ndarray) -> list[int]:
    extremes: list[int] = []
    for obj in (0, 1):
        extreme = min(
            range(int(normalized.shape[0])),
            key=lambda idx: (
                float(normalized[idx, obj]),
                float(ideal_distances[idx]),
                int(idx),
            ),
        )
        if extreme not in extremes:
            extremes.append(int(extreme))
    return extremes


def _resolve_selected_overflow(selected: list[int], slots: int, ideal_distances: np.ndarray) -> np.ndarray:
    chosen = sorted(selected, key=lambda idx: (float(ideal_distances[idx]), int(idx)))[:slots]
    return np.asarray(sorted(chosen), dtype=int)


def _selected_positions(order_positions: np.ndarray, selected: list[int]) -> list[int]:
    return sorted(int(order_positions[idx]) for idx in selected)


def _find_bracketing_selected(
    order: np.ndarray,
    order_positions: np.ndarray,
    selected_positions: list[int],
    candidate_idx: int,
) -> tuple[int, int] | None:
    if len(selected_positions) < 2:
        return None
    pos = int(order_positions[candidate_idx])
    insert_at = int(np.searchsorted(np.asarray(selected_positions, dtype=int), pos, side="left"))
    if insert_at <= 0 or insert_at >= len(selected_positions):
        return None
    return int(order[selected_positions[insert_at - 1]]), int(order[selected_positions[insert_at]])


def _point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    chord = end - start
    denom = float(np.dot(chord, chord))
    if denom <= 0.0:
        return 0.0
    t = float(np.dot(point - start, chord) / denom)
    t = min(1.0, max(0.0, t))
    projection = start + t * chord
    return float(np.linalg.norm(point - projection))


def _curvature_scores_2d(normalized: np.ndarray, order: np.ndarray) -> np.ndarray:
    curvatures = np.zeros(int(normalized.shape[0]), dtype=float)
    for pos, idx in enumerate(order.tolist()):
        if pos == 0 or pos == int(order.size) - 1:
            continue
        prev_idx = int(order[pos - 1])
        next_idx = int(order[pos + 1])
        curvatures[idx] = _point_to_segment_distance(normalized[idx], normalized[prev_idx], normalized[next_idx])
    return curvatures


def _gapfill_fallback_candidate(
    selected_set: set[int],
    ideal_distances: np.ndarray,
) -> int:
    return min(
        (idx for idx in range(int(ideal_distances.shape[0])) if idx not in selected_set),
        key=lambda idx: (float(ideal_distances[idx]), int(idx)),
    )


def _gapfill_valid_candidates(
    order: np.ndarray,
    order_positions: np.ndarray,
    selected_positions: list[int],
    selected_set: set[int],
    distances: np.ndarray,
) -> list[tuple[int, float, float]]:
    candidates: list[tuple[int, float, float]] = []
    for candidate_idx in order.tolist():
        if candidate_idx in selected_set:
            continue
        bracket = _find_bracketing_selected(order, order_positions, selected_positions, int(candidate_idx))
        if bracket is None:
            continue
        left_idx, right_idx = bracket
        d_lr = float(distances[left_idx, right_idx])
        d_left = float(distances[left_idx, candidate_idx])
        d_right = float(distances[candidate_idx, right_idx])
        gap_score = min(d_left, d_right) / (d_lr + _GAP_EPS)
        candidates.append((int(candidate_idx), float(gap_score), float(d_lr)))
    return candidates


def _select_split_front_gap_style(
    F_split: np.ndarray,
    slots: int,
    ideal: np.ndarray,
    nadir: np.ndarray,
    *,
    selector_name: str,
    use_curvature: bool,
) -> np.ndarray:
    points = np.asarray(F_split, dtype=float)
    n_points = int(points.shape[0])
    _require_two_objectives(points, selector_name)

    if slots >= n_points:
        return np.arange(n_points, dtype=int)
    if slots <= 0 or n_points == 0:
        return np.empty(0, dtype=int)

    normalized = _normalize_by_bounds(points, np.asarray(ideal, dtype=float), np.asarray(nadir, dtype=float))
    ideal_distances = _ideal_distances(normalized)
    order = _sorted_front_order_2d(normalized)
    order_positions = np.empty(n_points, dtype=int)
    order_positions[order] = np.arange(n_points, dtype=int)
    distances = _pairwise_distances(normalized)
    curvatures = _curvature_scores_2d(normalized, order) if use_curvature else None

    selected = _preserved_extremes_2d(normalized, ideal_distances)
    if len(selected) >= slots:
        return _resolve_selected_overflow(selected, slots, ideal_distances)

    selected_set = set(selected)
    while len(selected) < slots:
        selected_positions = _selected_positions(order_positions, selected)
        valid_candidates = _gapfill_valid_candidates(
            order,
            order_positions,
            selected_positions,
            selected_set,
            distances,
        )

        if not valid_candidates:
            best_idx = _gapfill_fallback_candidate(selected_set, ideal_distances)
        elif not use_curvature:
            best_idx = min(
                valid_candidates,
                key=lambda item: (
                    -float(item[1]),
                    -float(item[2]),
                    float(ideal_distances[int(item[0])]),
                    int(item[0]),
                ),
            )[0]
        else:
            gap_values = np.asarray([item[1] for item in valid_candidates], dtype=float)
            curv_values = np.asarray([float(curvatures[int(item[0])]) for item in valid_candidates], dtype=float)
            norm_gap = _normalize_scores(gap_values)
            norm_curv = _normalize_scores(curv_values)

            scored_candidates = [
                (
                    int(valid_candidates[pos][0]),
                    float(_CURVGAP_LAMBDA * norm_gap[pos] + (1.0 - _CURVGAP_LAMBDA) * norm_curv[pos]),
                    float(norm_gap[pos]),
                    float(norm_curv[pos]),
                )
                for pos in range(len(valid_candidates))
            ]
            best_idx = min(
                scored_candidates,
                key=lambda item: (
                    -float(item[1]),
                    -float(item[2]),
                    -float(item[3]),
                    float(ideal_distances[int(item[0])]),
                    int(item[0]),
                ),
            )[0]

        selected.append(int(best_idx))
        selected_set.add(int(best_idx))

    return np.asarray(sorted(selected), dtype=int)


def _build_complete_mst(distances: np.ndarray) -> list[tuple[int, int, float]]:
    """Return a deterministic Kruskal MST over the complete distance graph."""
    n_points = distances.shape[0]
    if n_points <= 1:
        return []

    tri_i, tri_j = np.triu_indices(n_points, k=1)
    weights = distances[tri_i, tri_j]
    order = np.lexsort((tri_j, tri_i, weights))

    parent = np.arange(n_points, dtype=int)
    rank = np.zeros(n_points, dtype=int)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return int(node)

    def union(a: int, b: int) -> bool:
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return False
        if rank[root_a] < rank[root_b]:
            parent[root_a] = root_b
        elif rank[root_a] > rank[root_b]:
            parent[root_b] = root_a
        else:
            parent[root_b] = root_a
            rank[root_a] += 1
        return True

    mst: list[tuple[int, int, float]] = []
    for edge_idx in order:
        i = int(tri_i[edge_idx])
        j = int(tri_j[edge_idx])
        w = float(weights[edge_idx])
        if union(i, j):
            mst.append((i, j, w))
            if len(mst) == n_points - 1:
                break
    return mst


def _component_weight(indices: np.ndarray, distances: np.ndarray) -> float:
    size = int(indices.size)
    if size <= 1:
        diameter = 0.0
    else:
        diameter = float(np.max(distances[np.ix_(indices, indices)]))
    return float(math.log1p(size) * diameter)


def _build_components(
    n_points: int,
    distances: np.ndarray,
    mst_edges: list[tuple[int, int, float]],
) -> list[_Component]:
    """Split the MST with the fixed median/MAD cut rule and return components."""
    if n_points == 0:
        return []
    if n_points == 1:
        indices = np.array([0], dtype=int)
        return [_Component(indices=indices, mst_edges=tuple(), weight=0.0)]

    edge_lengths = np.asarray([edge[2] for edge in mst_edges], dtype=float)
    median = float(np.median(edge_lengths))
    mad = float(np.median(np.abs(edge_lengths - median)))
    if mad == 0.0:
        cut_mask = edge_lengths > median
    else:
        cut_mask = edge_lengths > (median + 3.0 * mad)

    kept_edges = mst_edges if not np.any(cut_mask) else [edge for edge, cut in zip(mst_edges, cut_mask.tolist(), strict=True) if not cut]

    parent = np.arange(n_points, dtype=int)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return int(node)

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i, j, _w in kept_edges:
        union(i, j)

    groups: dict[int, list[int]] = {}
    for idx in range(n_points):
        groups.setdefault(find(idx), []).append(idx)

    edge_groups: dict[int, list[tuple[int, int]]] = {}
    for i, j, _w in kept_edges:
        root = find(i)
        edge_groups.setdefault(root, []).append((i, j))

    components: list[_Component] = []
    for root, members in sorted(groups.items(), key=lambda item: min(item[1])):
        indices = np.asarray(sorted(members), dtype=int)
        comp_edges = tuple(sorted(edge_groups.get(root, [])))
        components.append(
            _Component(
                indices=indices,
                mst_edges=comp_edges,
                weight=_component_weight(indices, distances),
            )
        )
    return components


def _build_single_component(
    n_points: int,
    distances: np.ndarray,
    mst_edges: list[tuple[int, int, float]],
) -> list[_Component]:
    if n_points == 0:
        return []
    indices = np.arange(n_points, dtype=int)
    return [
        _Component(
            indices=indices,
            mst_edges=tuple(sorted((int(i), int(j)) for i, j, _w in mst_edges)),
            weight=_component_weight(indices, distances),
        )
    ]


def _component_sort_key(component: _Component) -> tuple[float, int]:
    return (-component.weight, int(component.indices.min(initial=0)))


def _allocate_component_slots(components: list[_Component], slots: int) -> list[tuple[_Component, int]]:
    """
    Assign survivor counts to detected components.

    Each kept component receives at least one slot. If the component weights sum
    to zero, remaining slots are assigned deterministically by component size and
    then by the smallest split-front index.
    """
    if not components or slots <= 0:
        return []

    n_components = len(components)
    if n_components > slots:
        kept = sorted(components, key=_component_sort_key)[:slots]
        return [(component, 1) for component in kept]

    allocations = np.ones(n_components, dtype=int)
    capacities = np.asarray([int(component.indices.size) - 1 for component in components], dtype=int)
    remaining = int(slots - n_components)
    if remaining <= 0:
        return list(zip(components, allocations.tolist(), strict=True))

    weights = np.asarray([component.weight for component in components], dtype=float)
    total_weight = float(np.sum(weights))
    if total_weight > 0.0:
        quotas = remaining * (weights / total_weight)
        base = np.minimum(np.floor(quotas).astype(int), capacities)
        allocations += base
        capacities -= base
        remaining -= int(base.sum())

        remainders = quotas - np.floor(quotas)
        order = sorted(
            range(n_components),
            key=lambda idx: (-float(remainders[idx]), int(components[idx].indices.min(initial=0))),
        )
        while remaining > 0 and np.any(capacities > 0):
            assigned = False
            for idx in order:
                if capacities[idx] <= 0:
                    continue
                allocations[idx] += 1
                capacities[idx] -= 1
                remaining -= 1
                assigned = True
                if remaining == 0:
                    break
            if not assigned:
                break
    else:
        order = sorted(
            range(n_components),
            key=lambda idx: (-int(components[idx].indices.size), int(components[idx].indices.min(initial=0))),
        )
        while remaining > 0 and np.any(capacities > 0):
            assigned = False
            for idx in order:
                if capacities[idx] <= 0:
                    continue
                allocations[idx] += 1
                capacities[idx] -= 1
                remaining -= 1
                assigned = True
                if remaining == 0:
                    break
            if not assigned:
                break

    return list(zip(components, allocations.tolist(), strict=True))


def _all_pairs_shortest_paths(adjacency: np.ndarray) -> np.ndarray:
    dist = adjacency.copy()
    for mid in range(dist.shape[0]):
        dist = np.minimum(dist, dist[:, [mid]] + dist[[mid], :])
    return dist


def _select_within_component(
    component: _Component,
    n_keep: int,
    normalized: np.ndarray,
    distances: np.ndarray,
    ideal_distances: np.ndarray,
    *,
    distance_mode: str,
) -> np.ndarray:
    """
    Select survivors inside one component.

    The first seed is the point closest to the normalized ideal. Remaining
    survivors are added by farthest-first using either geodesic or Euclidean
    distances, with deterministic tie-breaking on ideal distance and then
    index.
    """
    indices = component.indices
    if n_keep >= indices.size:
        return indices.copy()
    if n_keep <= 0:
        return np.empty(0, dtype=int)

    comp_distances = distances[np.ix_(indices, indices)]
    comp_ideal = ideal_distances[indices]

    seed_local = min(range(indices.size), key=lambda idx: (float(comp_ideal[idx]), int(indices[idx])))
    selected_locals = [int(seed_local)]
    if n_keep == 1:
        return np.asarray([int(indices[seed_local])], dtype=int)

    n_component = int(indices.size)
    if distance_mode == "geodesic":
        max_neighbors = n_component - 1
        k_neighbors = min(max_neighbors, max(3, min(10, int(math.ceil(math.log2(n_component))))))

        adjacency = np.full((n_component, n_component), np.inf, dtype=float)
        np.fill_diagonal(adjacency, 0.0)

        if k_neighbors > 0:
            for row in range(n_component):
                order = np.argsort(comp_distances[row], kind="mergesort")
                neighbors = [int(idx) for idx in order if idx != row][:k_neighbors]
                for col in neighbors:
                    weight = float(comp_distances[row, col])
                    adjacency[row, col] = min(adjacency[row, col], weight)
                    adjacency[col, row] = min(adjacency[col, row], weight)

        local_pos = {int(global_idx): local_idx for local_idx, global_idx in enumerate(indices.tolist())}
        for i_global, j_global in component.mst_edges:
            i_local = local_pos[int(i_global)]
            j_local = local_pos[int(j_global)]
            weight = float(comp_distances[i_local, j_local])
            adjacency[i_local, j_local] = min(adjacency[i_local, j_local], weight)
            adjacency[j_local, i_local] = min(adjacency[j_local, i_local], weight)

        selection_distances = _all_pairs_shortest_paths(adjacency)
    elif distance_mode == "euclidean":
        selection_distances = comp_distances
    else:
        raise ValueError(f"Unsupported GCES distance mode: {distance_mode!r}")

    while len(selected_locals) < n_keep:
        selected_arr = np.asarray(selected_locals, dtype=int)
        candidates = [idx for idx in range(n_component) if idx not in selected_locals]
        candidate_arr = np.asarray(candidates, dtype=int)
        min_distances = np.min(selection_distances[np.ix_(candidate_arr, selected_arr)], axis=1)
        best_pos = min(
            range(candidate_arr.size),
            key=lambda pos: (
                -float(min_distances[pos]),
                float(comp_ideal[int(candidate_arr[pos])]),
                int(indices[int(candidate_arr[pos])]),
            ),
        )
        selected_locals.append(int(candidate_arr[best_pos]))

    return np.asarray(sorted(int(indices[idx]) for idx in selected_locals), dtype=int)


def _select_split_front(
    F_split: np.ndarray,
    slots: int,
    ideal: np.ndarray,
    nadir: np.ndarray,
    *,
    use_components: bool,
    distance_mode: str,
) -> np.ndarray:
    points = np.asarray(F_split, dtype=float)
    n_points = int(points.shape[0])

    if slots >= n_points:
        return np.arange(n_points, dtype=int)
    if slots <= 0 or n_points == 0:
        return np.empty(0, dtype=int)

    ideal_arr = np.asarray(ideal, dtype=float)
    nadir_arr = np.asarray(nadir, dtype=float)
    normalized = _normalize_by_bounds(points, ideal_arr, nadir_arr)

    distances = _pairwise_distances(normalized)
    mst_edges = _build_complete_mst(distances)
    if use_components:
        components = _build_components(n_points, distances, mst_edges)
        allocations = _allocate_component_slots(components, slots)
    else:
        components = _build_single_component(n_points, distances, mst_edges)
        allocations = [(components[0], slots)]

    ideal_distances = np.linalg.norm(normalized, axis=1)
    selected: list[int] = []
    for component, component_slots in allocations:
        selected.extend(
            _select_within_component(
                component,
                component_slots,
                normalized,
                distances,
                ideal_distances,
                distance_mode=distance_mode,
            ).tolist()
        )

    selected_arr = np.asarray(sorted(selected), dtype=int)
    if selected_arr.size != slots:
        raise ValueError("GCES selector returned an unexpected number of survivors.")
    return selected_arr


def select_split_front_nsga2_farthest(
    F_split: np.ndarray,
    slots: int,
    ideal: np.ndarray,
    nadir: np.ndarray,
    rng: Any,
) -> np.ndarray:
    """
    Select split-front survivors by deterministic farthest-first coverage.

    The selector first preserves one extreme per normalized objective whenever
    possible. If more extremes exist than available slots, they are resolved by
    smaller distance to the normalized ideal and then by local split-front
    index. No extra seed beyond the preserved extremes is added.
    """
    del rng
    points = np.asarray(F_split, dtype=float)
    n_points = int(points.shape[0])

    if slots >= n_points:
        return np.arange(n_points, dtype=int)
    if slots <= 0 or n_points == 0:
        return np.empty(0, dtype=int)

    normalized = _normalize_by_bounds(points, np.asarray(ideal, dtype=float), np.asarray(nadir, dtype=float))
    ideal_distances = np.linalg.norm(normalized, axis=1)
    n_obj = int(normalized.shape[1])

    selected: list[int] = []
    for obj in range(n_obj):
        extreme = min(
            range(n_points),
            key=lambda idx: (
                float(normalized[idx, obj]),
                float(ideal_distances[idx]),
                int(idx),
            ),
        )
        if extreme not in selected:
            selected.append(int(extreme))

    if len(selected) >= slots:
        chosen = sorted(selected, key=lambda idx: (float(ideal_distances[idx]), int(idx)))[:slots]
        return np.asarray(sorted(chosen), dtype=int)

    distances = _pairwise_distances(normalized)
    current_dmin = _initialize_current_dmin(distances, selected)
    candidate_mask = _initialize_candidate_mask(n_points, selected)
    while len(selected) < slots:
        candidate_arr = np.flatnonzero(candidate_mask)
        best_pos = min(
            range(candidate_arr.size),
            key=lambda idx: (
                -float(current_dmin[int(candidate_arr[idx])]),
                float(ideal_distances[int(candidate_arr[idx])]),
                int(candidate_arr[idx]),
            ),
        )
        best_idx = int(candidate_arr[best_pos])
        selected.append(best_idx)
        candidate_mask[best_idx] = False
        np.minimum(current_dmin, distances[:, best_idx], out=current_dmin)

    return np.asarray(sorted(selected), dtype=int)


def select_split_front_nsga2_gapfill(
    F_split: np.ndarray,
    slots: int,
    ideal: np.ndarray,
    nadir: np.ndarray,
    rng: Any,
) -> np.ndarray:
    """Select split-front survivors by deterministic 2D gap filling."""
    del rng
    return _select_split_front_gap_style(
        F_split,
        slots,
        ideal,
        nadir,
        selector_name="nsga2_gapfill",
        use_curvature=False,
    )


def select_split_front_nsga2_curvgap(
    F_split: np.ndarray,
    slots: int,
    ideal: np.ndarray,
    nadir: np.ndarray,
    rng: Any,
) -> np.ndarray:
    """Select split-front survivors by deterministic 2D curvature-aware gap filling."""
    del rng
    return _select_split_front_gap_style(
        F_split,
        slots,
        ideal,
        nadir,
        selector_name="nsga2_curvgap",
        use_curvature=True,
    )


def _require_objective_counts(
    points: np.ndarray,
    selector_name: str,
    *,
    allowed: tuple[int, ...],
) -> None:
    n_obj = int(points.shape[1]) if points.ndim == 2 else None
    if n_obj in allowed:
        return
    if allowed == (3,):
        raise ValueError(f"{selector_name} currently supports only 3-objective split fronts.")
    raise ValueError(f"{selector_name} currently supports only 2- or 3-objective split fronts.")


@cache
def _simplex_lattice(divisions: int, n_obj: int) -> np.ndarray:
    coords: list[tuple[int, ...]] = []

    def rec(remaining: int, depth: int, current: list[int]) -> None:
        if depth == n_obj - 1:
            current.append(remaining)
            coords.append(tuple(current))
            current.pop()
            return
        for value in range(remaining + 1):
            current.append(value)
            rec(remaining - value, depth + 1, current)
            current.pop()

    rec(int(divisions), 0, [])
    arr = np.asarray(coords, dtype=float)
    if arr.size == 0:
        raise ValueError("Failed to generate simplex lattice reference points.")
    arr /= float(divisions)
    arr = np.asarray(np.clip(arr, 0.0, 1.0), dtype=float)
    arr /= np.maximum(arr.sum(axis=1, keepdims=True), _GAP_EPS)
    arr.setflags(write=False)
    return np.asarray(arr, dtype=float)


def _reference_cover_points(n_obj: int) -> np.ndarray:
    if n_obj == 2:
        return _simplex_lattice(_REFCOVER_DIVISIONS_2D, n_obj)
    if n_obj == 3:
        return _simplex_lattice(_REFCOVER_DIVISIONS_3D, n_obj)
    raise ValueError("Reference-cover points are only defined for 2 or 3 objectives.")


@cache
def _sector_reference_directions_3d() -> np.ndarray:
    refs = _simplex_lattice(_SECTOR_DIVISIONS_3D, 3)
    norms = np.linalg.norm(refs, axis=1, keepdims=True)
    directions = refs / np.maximum(norms, _GAP_EPS)
    directions.setflags(write=False)
    return directions


def _preserved_extremes(normalized: np.ndarray, ideal_distances: np.ndarray) -> list[int]:
    selected: list[int] = []
    for obj in range(int(normalized.shape[1])):
        extreme = min(
            range(int(normalized.shape[0])),
            key=lambda idx: (
                float(normalized[idx, obj]),
                float(ideal_distances[idx]),
                int(idx),
            ),
        )
        if extreme not in selected:
            selected.append(int(extreme))
    return selected


def _unselected_candidates(n_points: int, selected_set: set[int]) -> np.ndarray:
    return np.asarray([idx for idx in range(n_points) if idx not in selected_set], dtype=int)


def _initialize_candidate_mask(n_points: int, selected: list[int]) -> np.ndarray:
    mask = np.ones(n_points, dtype=bool)
    if selected:
        mask[np.asarray(selected, dtype=int)] = False
    return mask


def _initialize_current_dmin(distances: np.ndarray, selected: list[int]) -> np.ndarray:
    selected_arr = np.asarray(selected, dtype=int)
    return np.asarray(np.min(distances[:, selected_arr], axis=1), dtype=float)


def _reference_distance_matrix(reference_points: np.ndarray, normalized: np.ndarray) -> np.ndarray:
    kernel = _get_reference_distance_matrix_jit()
    if kernel is not None:
        return np.asarray(
            kernel(
                np.asarray(reference_points, dtype=np.float64),
                np.asarray(normalized, dtype=np.float64),
            ),
            dtype=float,
        )
    diff = reference_points[:, None, :] - normalized[None, :, :]
    return np.asarray(np.linalg.norm(diff, axis=2), dtype=float)


def _get_reference_distance_matrix_jit() -> Any | None:
    global _REFERENCE_DISTANCE_MATRIX_JIT  # noqa: PLW0603
    if _REFERENCE_DISTANCE_MATRIX_JIT is not None:
        return _REFERENCE_DISTANCE_MATRIX_JIT
    njit = _get_numba_njit()
    if njit is None:
        return None

    @njit(cache=True)
    def _kernel(reference_points: np.ndarray, points: np.ndarray) -> np.ndarray:
        n_ref, n_obj = reference_points.shape
        n_points = points.shape[0]
        distances = np.empty((n_ref, n_points), dtype=np.float64)
        for ref_idx in range(n_ref):
            for point_idx in range(n_points):
                total = 0.0
                for obj in range(n_obj):
                    diff = reference_points[ref_idx, obj] - points[point_idx, obj]
                    total += diff * diff
                distances[ref_idx, point_idx] = math.sqrt(total)
        return distances

    _REFERENCE_DISTANCE_MATRIX_JIT = _kernel
    return _REFERENCE_DISTANCE_MATRIX_JIT


def _reference_cover_gains(
    current_ref_dist: np.ndarray,
    reference_distances: np.ndarray,
    candidate_arr: np.ndarray,
) -> np.ndarray:
    kernel = _get_reference_cover_gains_jit()
    if kernel is not None:
        return np.asarray(
            kernel(
                np.asarray(current_ref_dist, dtype=np.float64),
                np.asarray(reference_distances, dtype=np.float64),
                np.asarray(candidate_arr, dtype=np.int64),
            ),
            dtype=float,
        )

    improvements = np.maximum(current_ref_dist[:, None] - reference_distances[:, candidate_arr], 0.0)
    return np.asarray(np.mean(improvements, axis=0), dtype=float)


def _get_reference_cover_gains_jit() -> Any | None:
    global _REFERENCE_COVER_GAINS_JIT  # noqa: PLW0603
    if _REFERENCE_COVER_GAINS_JIT is not None:
        return _REFERENCE_COVER_GAINS_JIT
    njit = _get_numba_njit()
    if njit is None:
        return None

    @njit(cache=True)
    def _kernel(current_ref_dist: np.ndarray, reference_distances: np.ndarray, candidate_arr: np.ndarray) -> np.ndarray:
        n_ref = current_ref_dist.shape[0]
        n_candidates = candidate_arr.shape[0]
        gains = np.empty(n_candidates, dtype=np.float64)
        inv_n_ref = 0.0 if n_ref == 0 else 1.0 / float(n_ref)
        for pos in range(n_candidates):
            idx = candidate_arr[pos]
            total = 0.0
            for ref_idx in range(n_ref):
                improvement = current_ref_dist[ref_idx] - reference_distances[ref_idx, idx]
                if improvement > 0.0:
                    total += improvement
            gains[pos] = total * inv_n_ref
        return gains

    _REFERENCE_COVER_GAINS_JIT = _kernel
    return _REFERENCE_COVER_GAINS_JIT


def _candidate_dmin(current_dmin: np.ndarray, candidate_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dmin = np.asarray(current_dmin[candidate_arr], dtype=float)
    return dmin, _normalize_scores(dmin)


def _candidate_hv_gain(
    normalized: np.ndarray,
    selected_points: np.ndarray,
    selected_count: int,
    candidate_arr: np.ndarray,
    ref_point: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    current_hv = float(_hypervolume(selected_points[:selected_count], ref_point, allow_ref_expand=False))
    gains = np.empty(candidate_arr.size, dtype=float)
    for pos in range(candidate_arr.size):
        idx = int(candidate_arr[pos])
        selected_points[selected_count] = normalized[idx]
        trial = selected_points[: selected_count + 1]
        gains[pos] = max(float(_hypervolume(trial, ref_point, allow_ref_expand=False)) - current_hv, 0.0)
    return gains, _normalize_scores(gains)


def _candidate_reference_cover_gain(
    current_ref_dist: np.ndarray,
    reference_distances: np.ndarray,
    candidate_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    gains = _reference_cover_gains(current_ref_dist, reference_distances, candidate_arr)
    return gains, _normalize_scores(gains)


def _sector_assignments(
    normalized: np.ndarray,
    sector_dirs: np.ndarray,
) -> np.ndarray:
    assignments = np.full(normalized.shape[0], -1, dtype=int)
    norms = np.linalg.norm(normalized, axis=1)
    nonzero = norms > 0.0
    if not np.any(nonzero):
        return assignments
    unit = normalized[nonzero] / norms[nonzero, None]
    cosine = unit @ sector_dirs.T
    assignments[nonzero] = np.argmax(cosine, axis=1)
    return assignments


def _candidate_sector_rarity(
    sector_counts: np.ndarray,
    candidate_arr: np.ndarray,
    sector_assignments: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sectors = sector_assignments[candidate_arr] + 1
    rarity = np.asarray(1.0 / (1.0 + sector_counts[sectors]), dtype=float)
    return rarity, _normalize_scores(rarity)


def _finalize_farthest_family_selection(
    selected: list[int],
    slots: int,
    ideal_distances: np.ndarray,
) -> np.ndarray:
    if len(selected) >= slots:
        return _resolve_selected_overflow(selected, slots, ideal_distances)
    return np.asarray(sorted(selected), dtype=int)


def select_split_front_nsga2_hvfarthest(
    F_split: np.ndarray,
    slots: int,
    ideal: np.ndarray,
    nadir: np.ndarray,
    rng: Any,
) -> np.ndarray:
    """Select split-front survivors by farthest-first plus hypervolume gain."""
    del rng
    points = np.asarray(F_split, dtype=float)
    n_points = int(points.shape[0])
    _require_objective_counts(points, "nsga2_hvfarthest", allowed=(2, 3))

    if slots >= n_points:
        return np.arange(n_points, dtype=int)
    if slots <= 0 or n_points == 0:
        return np.empty(0, dtype=int)

    normalized = _normalize_by_bounds(points, np.asarray(ideal, dtype=float), np.asarray(nadir, dtype=float))
    ideal_distances = _ideal_distances(normalized)
    selected = _preserved_extremes(normalized, ideal_distances)
    if len(selected) >= slots:
        return _resolve_selected_overflow(selected, slots, ideal_distances)

    distances = _pairwise_distances(normalized)
    current_dmin = _initialize_current_dmin(distances, selected)
    candidate_mask = _initialize_candidate_mask(n_points, selected)
    n_obj = int(normalized.shape[1])
    ref_point = np.full(n_obj, _HV_REF_POINT_VALUE, dtype=float)
    selected_buffer = np.empty((slots + 1, n_obj), dtype=float)
    initial_selected = normalized[np.asarray(selected, dtype=int)]
    selected_buffer[: initial_selected.shape[0]] = initial_selected
    while len(selected) < slots:
        candidate_arr = np.flatnonzero(candidate_mask)
        dmin, norm_dmin = _candidate_dmin(current_dmin, candidate_arr)
        _hv_gain, norm_hv_gain = _candidate_hv_gain(
            normalized,
            selected_buffer,
            len(selected),
            candidate_arr,
            ref_point,
        )
        combined = 0.5 * norm_dmin + 0.5 * norm_hv_gain
        best_pos = min(
            range(candidate_arr.size),
            key=lambda pos: (
                -float(combined[pos]),
                -float(norm_dmin[pos]),
                -float(norm_hv_gain[pos]),
                float(ideal_distances[int(candidate_arr[pos])]),
                int(candidate_arr[pos]),
            ),
        )
        best_idx = int(candidate_arr[best_pos])
        selected.append(best_idx)
        candidate_mask[best_idx] = False
        np.minimum(current_dmin, distances[:, best_idx], out=current_dmin)
        selected_buffer[len(selected) - 1] = normalized[best_idx]

    return _finalize_farthest_family_selection(selected, slots, ideal_distances)


def select_split_front_nsga2_refcover_farthest(
    F_split: np.ndarray,
    slots: int,
    ideal: np.ndarray,
    nadir: np.ndarray,
    rng: Any,
) -> np.ndarray:
    """Select split-front survivors by farthest-first plus reference-cover gain."""
    del rng
    points = np.asarray(F_split, dtype=float)
    n_points = int(points.shape[0])
    _require_objective_counts(points, "nsga2_refcover_farthest", allowed=(2, 3))

    if slots >= n_points:
        return np.arange(n_points, dtype=int)
    if slots <= 0 or n_points == 0:
        return np.empty(0, dtype=int)

    normalized = _normalize_by_bounds(points, np.asarray(ideal, dtype=float), np.asarray(nadir, dtype=float))
    ideal_distances = _ideal_distances(normalized)
    selected = _preserved_extremes(normalized, ideal_distances)
    if len(selected) >= slots:
        return _resolve_selected_overflow(selected, slots, ideal_distances)

    ref_points = _reference_cover_points(int(normalized.shape[1]))
    reference_distances = _reference_distance_matrix(ref_points, normalized)
    distances = _pairwise_distances(normalized)
    current_dmin = _initialize_current_dmin(distances, selected)
    current_ref_dist = np.min(reference_distances[:, np.asarray(selected, dtype=int)], axis=1)
    candidate_mask = _initialize_candidate_mask(n_points, selected)
    while len(selected) < slots:
        candidate_arr = np.flatnonzero(candidate_mask)
        dmin, norm_dmin = _candidate_dmin(current_dmin, candidate_arr)
        _cover_gain, norm_cover_gain = _candidate_reference_cover_gain(current_ref_dist, reference_distances, candidate_arr)
        combined = 0.5 * norm_dmin + 0.5 * norm_cover_gain
        best_pos = min(
            range(candidate_arr.size),
            key=lambda pos: (
                -float(combined[pos]),
                -float(norm_dmin[pos]),
                -float(norm_cover_gain[pos]),
                float(ideal_distances[int(candidate_arr[pos])]),
                int(candidate_arr[pos]),
            ),
        )
        best_idx = int(candidate_arr[best_pos])
        selected.append(best_idx)
        candidate_mask[best_idx] = False
        np.minimum(current_dmin, distances[:, best_idx], out=current_dmin)
        np.minimum(current_ref_dist, reference_distances[:, best_idx], out=current_ref_dist)

    return _finalize_farthest_family_selection(selected, slots, ideal_distances)


def select_split_front_nsga2_hvref_farthest(
    F_split: np.ndarray,
    slots: int,
    ideal: np.ndarray,
    nadir: np.ndarray,
    rng: Any,
) -> np.ndarray:
    """Select split-front survivors by farthest-first plus HV and reference cover."""
    del rng
    points = np.asarray(F_split, dtype=float)
    n_points = int(points.shape[0])
    _require_objective_counts(points, "nsga2_hvref_farthest", allowed=(2, 3))

    if slots >= n_points:
        return np.arange(n_points, dtype=int)
    if slots <= 0 or n_points == 0:
        return np.empty(0, dtype=int)

    normalized = _normalize_by_bounds(points, np.asarray(ideal, dtype=float), np.asarray(nadir, dtype=float))
    ideal_distances = _ideal_distances(normalized)
    selected = _preserved_extremes(normalized, ideal_distances)
    if len(selected) >= slots:
        return _resolve_selected_overflow(selected, slots, ideal_distances)

    ref_points = _reference_cover_points(int(normalized.shape[1]))
    reference_distances = _reference_distance_matrix(ref_points, normalized)
    distances = _pairwise_distances(normalized)
    current_dmin = _initialize_current_dmin(distances, selected)
    current_ref_dist = np.min(reference_distances[:, np.asarray(selected, dtype=int)], axis=1)
    candidate_mask = _initialize_candidate_mask(n_points, selected)
    n_obj = int(normalized.shape[1])
    ref_point = np.full(n_obj, _HV_REF_POINT_VALUE, dtype=float)
    selected_buffer = np.empty((slots + 1, n_obj), dtype=float)
    initial_selected = normalized[np.asarray(selected, dtype=int)]
    selected_buffer[: initial_selected.shape[0]] = initial_selected
    while len(selected) < slots:
        candidate_arr = np.flatnonzero(candidate_mask)
        dmin, norm_dmin = _candidate_dmin(current_dmin, candidate_arr)
        _hv_gain, norm_hv_gain = _candidate_hv_gain(
            normalized,
            selected_buffer,
            len(selected),
            candidate_arr,
            ref_point,
        )
        _cover_gain, norm_cover_gain = _candidate_reference_cover_gain(current_ref_dist, reference_distances, candidate_arr)
        combined = 0.4 * norm_dmin + 0.3 * norm_hv_gain + 0.3 * norm_cover_gain
        best_pos = min(
            range(candidate_arr.size),
            key=lambda pos: (
                -float(combined[pos]),
                -float(norm_dmin[pos]),
                -float(norm_hv_gain[pos]),
                -float(norm_cover_gain[pos]),
                float(ideal_distances[int(candidate_arr[pos])]),
                int(candidate_arr[pos]),
            ),
        )
        best_idx = int(candidate_arr[best_pos])
        selected.append(best_idx)
        candidate_mask[best_idx] = False
        np.minimum(current_dmin, distances[:, best_idx], out=current_dmin)
        np.minimum(current_ref_dist, reference_distances[:, best_idx], out=current_ref_dist)
        selected_buffer[len(selected) - 1] = normalized[best_idx]

    return _finalize_farthest_family_selection(selected, slots, ideal_distances)


def select_split_front_nsga2_sector_farthest(
    F_split: np.ndarray,
    slots: int,
    ideal: np.ndarray,
    nadir: np.ndarray,
    rng: Any,
) -> np.ndarray:
    """Select split-front survivors by farthest-first plus 3D sector rarity."""
    del rng
    points = np.asarray(F_split, dtype=float)
    n_points = int(points.shape[0])
    _require_objective_counts(points, "nsga2_sector_farthest", allowed=(3,))

    if slots >= n_points:
        return np.arange(n_points, dtype=int)
    if slots <= 0 or n_points == 0:
        return np.empty(0, dtype=int)

    normalized = _normalize_by_bounds(points, np.asarray(ideal, dtype=float), np.asarray(nadir, dtype=float))
    ideal_distances = _ideal_distances(normalized)
    selected = _preserved_extremes(normalized, ideal_distances)
    if len(selected) >= slots:
        return _resolve_selected_overflow(selected, slots, ideal_distances)

    sector_dirs = _sector_reference_directions_3d()
    sector_assignments = _sector_assignments(normalized, sector_dirs)
    distances = _pairwise_distances(normalized)
    current_dmin = _initialize_current_dmin(distances, selected)
    candidate_mask = _initialize_candidate_mask(n_points, selected)
    sector_counts = np.zeros(int(sector_dirs.shape[0]) + 1, dtype=int)
    for idx in selected:
        sector_counts[int(sector_assignments[int(idx)]) + 1] += 1
    while len(selected) < slots:
        candidate_arr = np.flatnonzero(candidate_mask)
        dmin, norm_dmin = _candidate_dmin(current_dmin, candidate_arr)
        _rarity, norm_rarity = _candidate_sector_rarity(sector_counts, candidate_arr, sector_assignments)
        combined = 0.7 * norm_dmin + 0.3 * norm_rarity
        best_pos = min(
            range(candidate_arr.size),
            key=lambda pos: (
                -float(combined[pos]),
                -float(norm_dmin[pos]),
                -float(norm_rarity[pos]),
                float(ideal_distances[int(candidate_arr[pos])]),
                int(candidate_arr[pos]),
            ),
        )
        best_idx = int(candidate_arr[best_pos])
        selected.append(best_idx)
        candidate_mask[best_idx] = False
        np.minimum(current_dmin, distances[:, best_idx], out=current_dmin)
        sector_counts[int(sector_assignments[best_idx]) + 1] += 1

    return _finalize_farthest_family_selection(selected, slots, ideal_distances)


def select_split_front_gces(
    F_split: np.ndarray,
    slots: int,
    ideal: np.ndarray,
    nadir: np.ndarray,
    rng: Any,
) -> np.ndarray:
    """
    Select surviving local indices from an NSGA-II split front using GCES.

    The selector is deterministic by index order. ``rng`` is accepted only to
    keep the interface aligned with the algorithm host.
    """
    del rng
    return _select_split_front(
        F_split,
        slots,
        ideal,
        nadir,
        use_components=True,
        distance_mode="geodesic",
    )


def select_split_front_gces_nocomp(
    F_split: np.ndarray,
    slots: int,
    ideal: np.ndarray,
    nadir: np.ndarray,
    rng: Any,
) -> np.ndarray:
    """GCES ablation with component detection disabled."""
    del rng
    return _select_split_front(
        F_split,
        slots,
        ideal,
        nadir,
        use_components=False,
        distance_mode="geodesic",
    )


def select_split_front_gces_nogeo(
    F_split: np.ndarray,
    slots: int,
    ideal: np.ndarray,
    nadir: np.ndarray,
    rng: Any,
) -> np.ndarray:
    """GCES ablation with Euclidean farthest-first inside components."""
    del rng
    return _select_split_front(
        F_split,
        slots,
        ideal,
        nadir,
        use_components=True,
        distance_mode="euclidean",
    )


__all__ = [
    "select_split_front_gces",
    "select_split_front_gces_nocomp",
    "select_split_front_gces_nogeo",
    "select_split_front_nsga2_farthest",
    "select_split_front_nsga2_gapfill",
    "select_split_front_nsga2_curvgap",
    "select_split_front_nsga2_hvfarthest",
    "select_split_front_nsga2_refcover_farthest",
    "select_split_front_nsga2_hvref_farthest",
    "select_split_front_nsga2_sector_farthest",
]
