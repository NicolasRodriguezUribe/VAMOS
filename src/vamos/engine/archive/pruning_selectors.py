from __future__ import annotations

from math import comb

import numpy as np


def _count_lattice_points(n_obj: int, divisions: int) -> int:
    return comb(divisions + n_obj - 1, n_obj - 1)


def _choose_min_divisions(pop_size: int, n_obj: int) -> int:
    divisions = 1
    while _count_lattice_points(n_obj, divisions) < pop_size:
        divisions += 1
    return divisions


def _simplex_lattice(n_obj: int, divisions: int, limit: int | None) -> np.ndarray:
    coords: list[tuple[int, ...]] = []

    def rec(remaining: int, depth: int, current: list[int]) -> None:
        if limit is not None and len(coords) >= limit:
            return
        if depth == n_obj - 1:
            current.append(remaining)
            coords.append(tuple(current))
            current.pop()
            return
        for value in range(remaining + 1):
            current.append(value)
            rec(remaining - value, depth + 1, current)
            current.pop()
            if limit is not None and len(coords) >= limit:
                return

    rec(divisions, 0, [])
    arr = np.asarray(coords, dtype=float)
    arr /= divisions
    row_sums = arr.sum(axis=1, keepdims=True)
    arr = np.divide(arr, np.where(row_sums > 0.0, row_sums, 1.0), out=np.zeros_like(arr), where=row_sums > 0.0)
    return arr


def _generate_reference_directions(pop_size: int, n_obj: int) -> np.ndarray:
    if n_obj < 2:
        return np.ones((pop_size, 1), dtype=float)
    divisions = _choose_min_divisions(pop_size, n_obj)
    return _simplex_lattice(n_obj, divisions, limit=pop_size)


def _get_extreme_points(F: np.ndarray, ideal: np.ndarray) -> np.ndarray:
    if F.size == 0:
        return np.empty((0, F.shape[1] if F.ndim == 2 else 0), dtype=float)
    shifted = F - ideal
    shifted[shifted < 1e-3] = 0.0
    n_obj = F.shape[1]
    asf = np.eye(n_obj)
    asf[asf == 0] = 1e6
    vals = np.max(shifted * asf[:, None, :], axis=2)
    idx = np.argmin(vals, axis=1)
    return np.asarray(F[idx], dtype=float)


def _get_nadir_point(extreme_points: np.ndarray, ideal: np.ndarray, worst: np.ndarray) -> np.ndarray:
    try:
        plane = np.linalg.solve(extreme_points - ideal, np.ones(extreme_points.shape[1], dtype=float))
        intercepts = 1.0 / plane
        nadir = ideal + intercepts
        if np.any(intercepts <= 1e-6) or np.any(nadir > worst):
            raise np.linalg.LinAlgError
    except Exception:
        nadir = worst

    mask = nadir - ideal <= 1e-6
    nadir = nadir.copy()
    nadir[mask] = worst[mask]
    return nadir


def _associate(normalized_F: np.ndarray, ref_dirs_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(normalized_F, axis=1, keepdims=True)
    norms = np.where(norms > 0.0, norms, 1e-12)
    normalized_vectors = normalized_F / norms
    cosine = normalized_vectors @ ref_dirs_norm.T
    cosine = np.clip(cosine, -1.0, 1.0)
    associations = np.argmax(cosine, axis=1)
    cos_selected = cosine[np.arange(cosine.shape[0]), associations]
    distances = norms.flatten() * np.sqrt(1.0 - np.square(cos_selected))
    return associations, distances


def _niche_selection(
    front: np.ndarray,
    n_remaining: int,
    niche_counts: np.ndarray,
    associations: np.ndarray,
    distances: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    selected: list[int] = []
    if front.size == 0 or n_remaining <= 0:
        return np.empty(0, dtype=int)

    mask = np.ones(front.size, dtype=bool)
    while len(selected) < n_remaining:
        remaining = n_remaining - len(selected)
        active_pos = np.where(mask)[0]
        if active_pos.size == 0:
            break

        assoc_front = associations[front[active_pos]]
        unique_refs = np.unique(assoc_front)
        ref_counts = niche_counts[unique_refs]
        min_count = np.min(ref_counts)
        candidate_refs = unique_refs[ref_counts == min_count]
        rng.shuffle(candidate_refs)
        candidate_refs = candidate_refs[:remaining]

        for ref_choice in candidate_refs:
            candidates = active_pos[assoc_front == ref_choice]
            if candidates.size == 0:
                niche_counts[ref_choice] = np.inf
                continue
            rng.shuffle(candidates)
            if niche_counts[ref_choice] == 0:
                cand_dist = distances[front[candidates]]
                pick = candidates[int(np.argmin(cand_dist))]
            else:
                pick = candidates[0]
            mask[pick] = False
            selected.append(int(front[pick]))
            niche_counts[ref_choice] += 1
            if len(selected) >= n_remaining:
                break

    return np.asarray(selected, dtype=int)


def normalize_for_ref_dirs(F: np.ndarray) -> np.ndarray:
    """Normalize objective values for reference-direction association."""
    data = np.asarray(F, dtype=float)
    if data.ndim != 2:
        raise ValueError("F must be a 2D array.")
    if data.shape[0] == 0:
        return np.empty_like(data)

    ideal = data.min(axis=0)
    worst = data.max(axis=0)
    extreme_points = _get_extreme_points(data, ideal)
    nadir = _get_nadir_point(extreme_points, ideal, worst)

    denom = nadir - ideal
    fallback = worst - ideal
    denom = np.where(denom > 1e-12, denom, fallback)
    denom = np.where(denom > 1e-12, denom, 1.0)

    normalized = (data - ideal) / denom
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized


def select_maxmin_subset(F: np.ndarray, target_size: int) -> np.ndarray:
    """Select a diverse subset by greedy max-min distance in normalized objective space."""
    data = np.asarray(F, dtype=float)
    n = int(data.shape[0])
    if target_size <= 0:
        return np.empty(0, dtype=int)
    if n <= target_size:
        return np.arange(n, dtype=int)
    if target_size == 1:
        normalized = normalize_for_ref_dirs(data)
        best = int(np.argmin(np.sum(normalized, axis=1)))
        return np.asarray([best], dtype=int)

    normalized = normalize_for_ref_dirs(data)
    dist = np.linalg.norm(normalized[:, None, :] - normalized[None, :, :], axis=2)
    np.fill_diagonal(dist, -np.inf)

    first_pair = np.unravel_index(int(np.argmax(dist)), dist.shape)
    selected: list[int] = [int(first_pair[0]), int(first_pair[1])]
    if selected[0] == selected[1]:
        selected = [selected[0]]

    while len(selected) < target_size:
        remaining_mask = np.ones(n, dtype=bool)
        remaining_mask[np.asarray(selected, dtype=int)] = False
        remaining = np.flatnonzero(remaining_mask)
        if remaining.size == 0:
            break

        min_dist = dist[np.ix_(remaining, np.asarray(selected, dtype=int))].min(axis=1)
        best_value = float(np.max(min_dist))
        best_candidates = remaining[np.isclose(min_dist, best_value)]
        if best_candidates.size > 1:
            candidate_scores = np.sum(normalized[best_candidates], axis=1)
            best_idx = int(best_candidates[np.argmin(candidate_scores)])
        else:
            best_idx = int(best_candidates[0])
        selected.append(best_idx)

    return np.asarray(selected[:target_size], dtype=int)


def select_ref_dirs_subset(
    F: np.ndarray,
    target_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select a subset by NSGA-III-style reference-direction niching."""
    data = np.asarray(F, dtype=float)
    n = int(data.shape[0])
    if target_size <= 0:
        return np.empty(0, dtype=int)
    if n <= target_size:
        return np.arange(n, dtype=int)

    ref_dirs = _generate_reference_directions(target_size, data.shape[1])
    ref_norms = np.linalg.norm(ref_dirs, axis=1, keepdims=True)
    ref_dirs_norm = np.divide(
        ref_dirs,
        np.where(ref_norms > 0.0, ref_norms, 1.0),
        out=np.zeros_like(ref_dirs),
        where=ref_norms > 0.0,
    )

    normalized = normalize_for_ref_dirs(data)
    associations, distances = _associate(normalized, ref_dirs_norm)
    selected = _niche_selection(
        np.arange(n, dtype=int),
        target_size,
        np.zeros(ref_dirs_norm.shape[0], dtype=int),
        associations,
        distances,
        rng,
    )
    if selected.size >= target_size:
        return np.asarray(selected[:target_size], dtype=int)

    remaining_mask = np.ones(n, dtype=bool)
    remaining_mask[selected] = False
    remaining = np.flatnonzero(remaining_mask)
    if remaining.size == 0:
        return np.asarray(selected, dtype=int)

    fill_local = select_maxmin_subset(data[remaining], target_size - selected.size)
    fill_global = remaining[fill_local]
    return np.concatenate([np.asarray(selected, dtype=int), np.asarray(fill_global, dtype=int)])
