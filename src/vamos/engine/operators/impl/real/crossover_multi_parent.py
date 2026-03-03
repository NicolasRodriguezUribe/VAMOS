"""Multi-parent real-valued crossover operators."""

from __future__ import annotations

import numpy as np

from ._crossover_base import Crossover
from .utils import ArrayLike, _ensure_bounds


class PCXCrossover(Crossover):
    """Parent-Centric Crossover (PCX) using 3-parent groups."""

    def __init__(
        self,
        sigma_eta: float = 0.1,
        sigma_zeta: float = 0.1,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
        prob_crossover: float | None = None,
        workspace: object | None = None,
        allow_inplace: bool = False,
    ) -> None:
        _ = prob_crossover, workspace, allow_inplace
        self.sigma_eta = float(sigma_eta)
        self.sigma_zeta = float(sigma_zeta)
        self.lower, self.upper = _ensure_bounds(lower, upper)

    def __call__(self, parents: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        groups = self._as_matings(parents, expected_parents=3, copy=False)
        n_groups, _, n_vars = groups.shape
        offspring = np.empty_like(groups)
        for i in range(n_groups):
            p = groups[i]
            x0 = p[0]
            others = p[1:]
            centroid = np.mean(others, axis=0)
            d = centroid - x0
            diff = others - x0
            basis = np.linalg.qr(diff.T, mode="reduced")[0] if diff.shape[0] > 0 else np.eye(n_vars)
            avg_dist = np.mean(np.linalg.norm(diff, axis=1)) if diff.size else 0.0
            for j in range(3):
                noise = rng.normal(0.0, self.sigma_zeta * (avg_dist or 1.0), size=basis.shape[1])
                offspring[i, j, :] = x0 + self.sigma_eta * d + (basis @ noise)
        return offspring


class UNDXCrossover(Crossover):
    """Unimodal Normal Distribution Crossover (UNDX)."""

    def __init__(
        self,
        prob_crossover: float = 0.9,
        zeta: float = 0.5,
        eta: float = 0.35,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
        workspace: object | None = None,
        allow_inplace: bool = False,
    ) -> None:
        _ = workspace, allow_inplace
        self.prob = float(prob_crossover)
        self.zeta = float(zeta)
        self.eta = float(eta)
        self.lower, self.upper = _ensure_bounds(lower, upper)

    def __call__(self, parents: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        groups = self._as_matings(parents, expected_parents=3, copy=False)
        n_groups, _, n_vars = groups.shape
        if n_groups == 0:
            return np.empty((0, 2, n_vars), dtype=groups.dtype)
        offspring = np.empty((n_groups, 2, n_vars), dtype=groups.dtype)
        for i in range(n_groups):
            p1, p2, p3 = groups[i]
            if rng.random() > self.prob:
                offspring[i, 0, :] = p1
                offspring[i, 1, :] = p2
                continue
            center = 0.5 * (p1 + p2)
            diff = p2 - p1
            distance = np.linalg.norm(diff)
            if distance < 1.0e-10:
                offspring[i, 0, :] = p1
                offspring[i, 1, :] = p2
                continue
            child1 = np.empty(n_vars, dtype=groups.dtype)
            child2 = np.empty(n_vars, dtype=groups.dtype)
            for j in range(n_vars):
                alpha = rng.uniform(-self.zeta * distance, self.zeta * distance)
                beta = (rng.random() - 0.5) * self.eta * distance + (rng.random() - 0.5) * self.eta * distance
                orthogonal = (p3[j] - center[j]) / distance
                child1[j] = center[j] + alpha * diff[j] / distance + beta * orthogonal
                child2[j] = center[j] - alpha * diff[j] / distance - beta * orthogonal
            offspring[i, 0, :] = child1
            offspring[i, 1, :] = child2
        return offspring


class SPXCrossover(Crossover):
    """Simplex crossover (SPX) sampling inside the simplex spanned by parents."""

    def __init__(
        self,
        epsilon: float = 0.5,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
        prob_crossover: float | None = None,
        workspace: object | None = None,
        allow_inplace: bool = False,
    ) -> None:
        _ = prob_crossover, workspace, allow_inplace
        self.epsilon = float(epsilon)
        self.lower, self.upper = _ensure_bounds(lower, upper)

    def __call__(self, parents: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        groups = self._as_matings(parents, expected_parents=3, copy=False)
        n_groups, k, _ = groups.shape
        offspring = np.empty_like(groups)
        for i in range(n_groups):
            group = groups[i]
            centroid = np.mean(group, axis=0)
            for j in range(k):
                weights = np.asarray(rng.random(k), dtype=float)
                weights /= float(weights.sum())
                point = np.sum(weights[:, None] * group, axis=0)
                offspring[i, j, :] = centroid + self.epsilon * (point - centroid)
        return offspring


__all__ = ["PCXCrossover", "SPXCrossover", "UNDXCrossover"]
