"""Simple directional real-valued intensification operator."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._intensification_base import IntensificationOperator
from .utils import ArrayLike, VariationWorkspace, _ensure_bounds


class DirectionalIntensification(IntensificationOperator):
    """Lightweight offspring refinement using decision-space attraction only.

    This operator is intentionally simpler than PAVE. Each active offspring
    vector ``x`` is updated as

        x' = x + alpha * d_n + beta * d_t

    where:
    - ``d_n`` points toward the centroid of nearby decision-space neighbors
    - ``d_t`` follows the direction induced by the parent pair when available

    The neighborhood pool prefers the bound workspace population and degrades to
    parent-only or offspring-only context when that population is unavailable.
    Bounds repair is intentionally left to :class:`VariationPipeline`.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        alpha: float = 0.25,
        beta: float = 0.1,
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
        self.k_neighbors = int(k_neighbors)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.prob = float(prob_intensification)
        self.lower, self.upper = _ensure_bounds(lower, upper)
        self.workspace = workspace
        self.allow_inplace = bool(allow_inplace)
        self.parents_per_group = int(parents_per_group)
        self.children_per_group = int(children_per_group)

    def __call__(
        self,
        offspring: ArrayLike,
        rng: np.random.Generator,
        *,
        parents: ArrayLike | None = None,
    ) -> ArrayLike:
        """Refine a flat offspring population with decision-space attraction."""
        offspring_arr = self._as_population(offspring, name="offspring", copy=False)
        refined = offspring_arr if self.allow_inplace else offspring_arr.copy()
        if refined.shape[0] == 0:
            return refined

        self._check_bounds_match(refined, self.lower)
        active_mask = self._sample_mask(refined.shape[0], rng)
        if not np.any(active_mask):
            return refined

        # Keep the source batch stable so local neighborhoods do not drift as
        # intensified solutions are written back in the current call.
        source_offspring = np.array(offspring_arr, copy=True)
        base_parents, mates = self._resolve_parent_context(source_offspring, parents)
        candidate_X = self._resolve_context(source_offspring, base_parents)

        for idx in np.flatnonzero(active_mask):
            refined[idx] = self._generate_candidate(
                source_offspring[idx],
                None if base_parents is None else base_parents[idx],
                None if mates is None else mates[idx],
                candidate_X,
            )
        return refined

    def _sample_mask(self, n_rows: int, rng: np.random.Generator) -> np.ndarray:
        """Return the per-offspring intensification mask."""
        if self.workspace is None:
            return rng.random(n_rows) <= self.prob
        probs = self._buffer("directional_intensify_prob", (n_rows,), np.float64)
        rng.random(out=probs)
        mask = self._buffer("directional_intensify_mask", (n_rows,), np.bool_)
        np.less_equal(probs, self.prob, out=mask)
        return mask

    def _buffer(self, key: str, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
        """Request a reusable scratch array when a workspace is available."""
        if self.workspace is None:
            return np.empty(shape, dtype=dtype)
        return self.workspace.request(key, shape, dtype)

    def _resolve_parent_context(
        self,
        offspring_arr: np.ndarray,
        parents: ArrayLike | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Recover base-parent and mate arrays when the flat mapping is valid."""
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
    ) -> np.ndarray:
        """Return the candidate neighborhood pool for centroid attraction."""
        workspace = self.workspace
        if workspace is not None and workspace.population is not None:
            population = np.asarray(workspace.population, dtype=float)
            if population.ndim == 2 and population.shape[1] == offspring_arr.shape[1]:
                return population
        if parents_arr is not None:
            return parents_arr
        return offspring_arr

    def _select_neighbors(self, x: np.ndarray, candidate_X: np.ndarray) -> np.ndarray:
        """Select the nearest decision-space neighbors, excluding self matches."""
        distances = np.linalg.norm(candidate_X - x, axis=1)
        is_self = np.all(np.isclose(candidate_X, x, rtol=1e-10, atol=1e-12), axis=1)
        distances = np.asarray(distances, dtype=float).copy()
        distances[is_self] = np.inf
        valid_indices = np.flatnonzero(np.isfinite(distances))
        if valid_indices.size == 0:
            return x[np.newaxis, :]

        order = valid_indices[np.argsort(distances[valid_indices], kind="mergesort")]
        k = min(self.k_neighbors, order.size)
        return candidate_X[order[:k]]

    def _centroid_direction(self, x: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
        """Compute the decision-space centroid attraction term."""
        if neighbors.ndim != 2 or neighbors.shape[0] == 0:
            return np.zeros_like(x)
        return np.mean(neighbors, axis=0) - x

    def _tangential_direction(
        self,
        base_parent: np.ndarray | None,
        mate: np.ndarray | None,
    ) -> np.ndarray:
        """Recover the pairwise directional term when parent context exists."""
        if base_parent is None or mate is None:
            return np.zeros_like(self.lower)
        return base_parent - mate

    def _generate_candidate(
        self,
        x: np.ndarray,
        base_parent: np.ndarray | None,
        mate: np.ndarray | None,
        candidate_X: np.ndarray,
    ) -> np.ndarray:
        """Generate one intensified offspring using local centroid attraction."""
        neighbors = self._select_neighbors(x, candidate_X)
        d_n = self._centroid_direction(x, neighbors)
        d_t = self._tangential_direction(base_parent, mate)
        return x + self.alpha * d_n + self.beta * d_t


__all__ = ["DirectionalIntensification"]
