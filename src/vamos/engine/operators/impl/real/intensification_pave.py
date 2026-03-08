"""Pareto-aware real-valued intensification operator."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._intensification_base import IntensificationOperator
from .utils import ArrayLike, VariationWorkspace, _ensure_bounds


class PAVEIntensification(IntensificationOperator):
    """Pareto-Aware Variation for Exploitation (PAVE).

    PAVE refines existing offspring using population-level context when it is
    available. For each offspring vector ``x`` it applies

        x' = x + alpha * d_n + beta * d_t

    where ``d_n`` is the direction from ``x`` to the centroid of nearby
    population members and ``d_t`` is a tangential direction recovered from the
    parent pair when a pairwise parent mapping is available.

    The operator prefers the current population and objective values bound in
    ``VariationWorkspace``. When that context is unavailable it degrades
    gracefully to parent-only neighborhoods and decision-space distances.
    Bounds repair is intentionally left to :class:`VariationPipeline`.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        alpha: float = 0.35,
        beta: float = 0.2,
        lambda_distance: float = 0.5,
        prob_intensification: float = 1.0,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
        workspace: VariationWorkspace | None = None,
        allow_inplace: bool = False,
        parents_per_group: int = 2,
        children_per_group: int = 2,
    ) -> None:
        if int(k_neighbors) <= 0:
            raise ValueError("k_neighbors must be positive.")
        if not 0.0 <= float(lambda_distance) <= 1.0:
            raise ValueError("lambda_distance must be in [0, 1].")
        self.k_neighbors = int(k_neighbors)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.lambda_distance = float(lambda_distance)
        self.prob = float(prob_intensification)
        self.lower, self.upper = _ensure_bounds(lower, upper)
        self.workspace = workspace
        self.allow_inplace = bool(allow_inplace)
        self.parents_per_group = int(parents_per_group)
        self.children_per_group = int(children_per_group)
        self.call_count = 0
        self.last_population_context_available = False
        self.last_objective_context_available = False
        self.last_fallback_used = True

    def __call__(
        self,
        offspring: ArrayLike,
        rng: np.random.Generator,
        *,
        parents: ArrayLike | None = None,
    ) -> ArrayLike:
        """Refine a flat offspring population with optional parent context."""
        self.call_count += 1
        offspring_arr = self._as_population(offspring, name="offspring", copy=False)
        refined = offspring_arr if self.allow_inplace else offspring_arr.copy()
        if refined.shape[0] == 0:
            return refined

        self._check_bounds_match(refined, self.lower)
        active_mask = self._sample_mask(refined.shape[0], rng)
        if not np.any(active_mask):
            return refined

        # Freeze the current batch so fallback neighborhoods do not drift as
        # intensified offspring are written back.
        source_offspring = np.array(offspring_arr, copy=True)
        base_parents, mates = self._resolve_parent_context(source_offspring, parents)
        candidate_X, candidate_F = self._resolve_context(source_offspring, base_parents)

        # Each active offspring is nudged toward a neighborhood centroid and
        # along a tangential direction recovered from its parent pair.
        for idx in np.flatnonzero(active_mask):
            x = refined[idx]
            base_parent = None if base_parents is None else base_parents[idx]
            mate = None if mates is None else mates[idx]
            refined[idx] = self._generate_child(x, base_parent, mate, candidate_X, candidate_F)
        return refined

    def _sample_mask(self, n_rows: int, rng: np.random.Generator) -> np.ndarray:
        """Return the per-offspring intensification mask."""
        if self.workspace is None:
            return rng.random(n_rows) <= self.prob
        probs = self._buffer("pave_intensify_prob", (n_rows,), np.float64)
        rng.random(out=probs)
        mask = self._buffer("pave_intensify_mask", (n_rows,), np.bool_)
        np.less_equal(probs, self.prob, out=mask)
        return mask

    def _buffer(self, key: str, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
        """Request a reusable scratch array when the workspace supports it."""
        if self.workspace is None:
            return np.empty(shape, dtype=dtype)
        return self.workspace.request(key, shape, dtype)

    def _resolve_parent_context(
        self,
        offspring_arr: np.ndarray,
        parents: ArrayLike | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Recover base-parent and mate arrays when pairwise mapping is available."""
        if parents is None or self.parents_per_group != 2 or self.children_per_group != 2:
            return None, None

        parents_arr = np.asarray(parents, dtype=float)
        if parents_arr.ndim == 3:
            if parents_arr.shape[1] < 2:
                return None, None
            parents_arr = parents_arr[:, :2, :].reshape(-1, parents_arr.shape[-1])

        if parents_arr.ndim != 2 or parents_arr.shape != offspring_arr.shape:
            return None, None
        if parents_arr.shape[0] % 2 != 0:
            return None, None

        mate_idx = np.arange(parents_arr.shape[0], dtype=int).reshape(-1, 2)[:, ::-1].reshape(-1)
        return parents_arr, parents_arr[mate_idx]

    def _resolve_context(
        self,
        offspring_arr: np.ndarray,
        parents_arr: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Return the neighborhood pool and aligned objective values when available."""
        workspace = self.workspace
        if workspace is not None and workspace.population is not None:
            population = np.asarray(workspace.population, dtype=float)
            if population.ndim == 2 and population.shape[1] == offspring_arr.shape[1]:
                self.last_population_context_available = True
                objectives = workspace.objectives
                if objectives is None:
                    self.last_objective_context_available = False
                    self.last_fallback_used = True
                    return population, None
                objective_arr = np.asarray(objectives, dtype=float)
                if objective_arr.ndim == 2 and objective_arr.shape[0] == population.shape[0]:
                    self.last_objective_context_available = True
                    self.last_fallback_used = False
                    return population, objective_arr
                self.last_objective_context_available = False
                self.last_fallback_used = True
                return population, None

        self.last_population_context_available = False
        self.last_objective_context_available = False
        self.last_fallback_used = True
        if parents_arr is not None:
            return parents_arr, None
        return offspring_arr, None

    def _match_objectives(
        self,
        base_parent: np.ndarray | None,
        population: np.ndarray,
        objectives: np.ndarray | None,
    ) -> np.ndarray | None:
        """Recover the parent objective vector used for objective-space scoring."""
        if base_parent is None or objectives is None or population.shape[0] != objectives.shape[0]:
            return None
        matches = np.all(np.isclose(population, base_parent, rtol=1e-10, atol=1e-12), axis=1)
        indices = np.flatnonzero(matches)
        if indices.size == 0:
            return None
        return objectives[int(indices[0])]

    @staticmethod
    def _normalize_distances(distances: np.ndarray) -> np.ndarray:
        """Normalize distances to [0, 1] before hybrid combination."""
        arr = np.asarray(distances, dtype=float)
        finite = np.isfinite(arr)
        if not np.any(finite):
            return np.zeros_like(arr, dtype=float)

        finite_vals = arr[finite]
        min_val = float(np.min(finite_vals))
        max_val = float(np.max(finite_vals))
        scale = max_val - min_val
        out = np.zeros_like(arr, dtype=float)
        if scale <= 1.0e-12:
            return out
        out[finite] = (finite_vals - min_val) / scale
        return out

    def _hybrid_scores(
        self,
        x: np.ndarray,
        base_obj: np.ndarray | None,
        candidate_X: np.ndarray,
        candidate_F: np.ndarray | None,
    ) -> np.ndarray:
        """Compute normalized hybrid scores for one focal offspring."""
        decision_distance = np.linalg.norm(candidate_X - x, axis=1)
        decision_score = self._normalize_distances(decision_distance)
        if base_obj is None or candidate_F is None or candidate_F.shape[0] != candidate_X.shape[0]:
            return decision_score

        objective_distance = np.linalg.norm(candidate_F - base_obj, axis=1)
        objective_score = self._normalize_distances(objective_distance)
        return self.lambda_distance * objective_score + (1.0 - self.lambda_distance) * decision_score

    def _select_neighbors(
        self,
        x: np.ndarray,
        candidate_X: np.ndarray,
        scores: np.ndarray,
    ) -> np.ndarray:
        """Select the k best neighbors, excluding exact self matches."""
        is_self = np.all(np.isclose(candidate_X, x, rtol=1e-10, atol=1e-12), axis=1)
        valid_scores = np.asarray(scores, dtype=float).copy()
        valid_scores[is_self] = np.inf
        valid_indices = np.flatnonzero(np.isfinite(valid_scores))
        if valid_indices.size == 0:
            return x[np.newaxis, :]

        order = valid_indices[np.argsort(valid_scores[valid_indices], kind="mergesort")]
        k = min(self.k_neighbors, order.size)
        return candidate_X[order[:k]]

    def _centroid_direction(self, x: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
        """Compute the neighborhood direction toward the centroid."""
        if neighbors.ndim != 2 or neighbors.shape[0] == 0:
            return np.zeros_like(x)
        centroid = np.mean(neighbors, axis=0)
        return centroid - x

    def _tangential_direction(
        self,
        base_parent: np.ndarray | None,
        mate: np.ndarray | None,
    ) -> np.ndarray:
        """Compute the parent-pair tangential direction when available."""
        if base_parent is None or mate is None:
            return np.zeros_like(self.lower)
        return base_parent - mate

    def _generate_child(
        self,
        x: np.ndarray,
        base_parent: np.ndarray | None,
        mate: np.ndarray | None,
        candidate_X: np.ndarray,
        candidate_F: np.ndarray | None,
    ) -> np.ndarray:
        """Generate one intensified offspring from the current offspring vector."""
        base_obj = self._match_objectives(base_parent, candidate_X, candidate_F)
        scores = self._hybrid_scores(x, base_obj, candidate_X, candidate_F)
        neighbors = self._select_neighbors(x, candidate_X, scores)
        d_n = self._centroid_direction(x, neighbors)
        d_t = self._tangential_direction(base_parent, mate)
        return x + self.alpha * d_n + self.beta * d_t


__all__ = ["PAVEIntensification"]
