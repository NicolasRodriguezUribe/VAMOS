"""Real-valued crossover operators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .utils import (
    ArrayLike,
    RealOperator,
    VariationWorkspace,
    _check_nvars,
    _clip_population,
    _ensure_bounds,
)


class Crossover(RealOperator, ABC):
    """Base class for real-coded crossover operators."""

    @abstractmethod
    def __call__(self, parents: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        raise NotImplementedError


class SBXCrossover(Crossover):
    """Simulated Binary Crossover (SBX) operator."""

    def __init__(
        self,
        prob_crossover: float = 0.9,
        eta: float = 10.0,
        prob_var: float = 0.5,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
        workspace: VariationWorkspace | None = None,
        allow_inplace: bool = False,
    ) -> None:
        self.prob = float(prob_crossover)
        self.eta = float(eta)
        self.prob_var = float(prob_var)
        self.lower, self.upper = _ensure_bounds(lower, upper)
        self.workspace = workspace
        self.allow_inplace = bool(allow_inplace)

    def _rand(self, key: str, shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
        if self.workspace is None:
            return rng.random(shape)
        buf = self.workspace.request(key, shape, np.float64)
        rng.random(out=buf)
        return buf

    def _mask_pairs(self, n_pairs: int, rng: np.random.Generator) -> np.ndarray:
        if self.workspace is None:
            return rng.random(n_pairs) <= self.prob
        probs = self.workspace.request("sbx_prob", (n_pairs,), np.float64)
        rng.random(out=probs)
        mask = self.workspace.request("sbx_mask", (n_pairs,), np.bool_)
        np.less_equal(probs, self.prob, out=mask)
        return mask

    def _buffer(self, key: str, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
        if self.workspace is None:
            return np.empty(shape, dtype=dtype)
        return self.workspace.request(key, shape, dtype)

    def __call__(self, parents: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        parents_arr = self._as_matings(parents, copy=False, name="parents")
        offspring = parents_arr if self.allow_inplace else parents_arr.copy()
        n_pairs, _, _ = offspring.shape
        if n_pairs == 0:
            return offspring
        self._check_bounds_match(offspring[:, 0, :], self.lower)

        pair_mask = self._mask_pairs(n_pairs, rng)
        idx = np.flatnonzero(pair_mask)
        if idx.size == 0:
            return offspring

        parent1 = offspring[idx, 0, :]
        parent2 = offspring[idx, 1, :]
        base1 = parent1.copy()
        base2 = parent2.copy()
        eps = 1.0e-14

        y1 = np.minimum(parent1, parent2)
        y2 = np.maximum(parent1, parent2)
        diff = y2 - y1
        valid = diff > eps
        if self.prob_var >= 1.0:
            active = valid
        else:
            var_rand = self._rand("sbx_var", parent1.shape, rng)
            if self.workspace is None:
                var_mask = var_rand <= self.prob_var
            else:
                var_mask = self.workspace.request("sbx_var_mask", parent1.shape, np.bool_)
                np.less_equal(var_rand, self.prob_var, out=var_mask)
            active = valid & var_mask
        if not np.any(active):
            return offspring

        xl = self.lower.reshape(1, -1)
        xu = self.upper.reshape(1, -1)
        rand = self._rand("sbx_rand", parent1.shape, rng)
        betaq = self._buffer("sbx_betaq", parent1.shape, parent1.dtype)

        beta_valid = 1.0 + (2.0 * (y1 - xl) / diff.clip(min=eps))
        beta_valid = np.maximum(beta_valid, eps)
        alpha = 2.0 - np.power(beta_valid, -(self.eta + 1.0))
        alpha = np.maximum(alpha, eps)
        term = rand <= (1.0 / alpha)
        inv_eta = 1.0 / (self.eta + 1.0)
        betaq[term] = np.power(rand[term] * alpha[term], inv_eta)
        betaq[~term] = np.power(1.0 / (2.0 - rand[~term] * alpha[~term]), inv_eta)
        c1 = 0.5 * ((y1 + y2) - betaq * diff)

        beta_valid = 1.0 + (2.0 * (xu - y2) / diff.clip(min=eps))
        beta_valid = np.maximum(beta_valid, eps)
        alpha = 2.0 - np.power(beta_valid, -(self.eta + 1.0))
        alpha = np.maximum(alpha, eps)
        term = rand <= (1.0 / alpha)
        betaq[term] = np.power(rand[term] * alpha[term], inv_eta)
        betaq[~term] = np.power(1.0 / (2.0 - rand[~term] * alpha[~term]), inv_eta)
        c2 = 0.5 * ((y1 + y2) + betaq * diff)

        np.clip(c1, self.lower, self.upper, out=c1)
        np.clip(c2, self.lower, self.upper, out=c2)
        swap = self._rand("sbx_swap", parent1.shape, rng)
        if self.workspace is None:
            swap_mask = swap <= 0.5
        else:
            swap_mask = self.workspace.request("sbx_swap_mask", parent1.shape, np.bool_)
            np.less_equal(swap, 0.5, out=swap_mask)
        child1 = np.where(active, c1, base1)
        child2 = np.where(active, c2, base2)
        swap_mask = swap_mask & active
        new_child1 = np.where(swap_mask, child2, child1)
        new_child2 = np.where(swap_mask, child1, child2)

        offspring[idx, 0, :] = new_child1
        offspring[idx, 1, :] = new_child2
        return offspring


class BLXAlphaCrossover(Crossover):
    """Blend crossover (BLX-alpha) with optional buffer reuse and repair strategies."""

    def __init__(
        self,
        alpha: float = 0.5,
        prob_crossover: float = 0.9,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
        repair: str = "clip",
        workspace: VariationWorkspace | None = None,
        allow_inplace: bool = False,
    ) -> None:
        self.alpha = float(alpha)
        self.prob = float(prob_crossover)
        self.lower, self.upper = _ensure_bounds(lower, upper)
        normalized = (repair or "clip").lower()
        if normalized == "random":
            normalized = "resample"
        if normalized not in {"clip", "resample", "reflect", "round"}:
            raise ValueError(f"Unsupported BLX repair strategy '{repair}'.")
        self.repair = normalized
        self.workspace = workspace
        self.allow_inplace = bool(allow_inplace)

    def _rand(self, key: str, shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
        if self.workspace is None:
            return rng.random(shape)
        buf = self.workspace.request(key, shape, np.float64)
        rng.random(out=buf)
        return buf

    def _sample_mask(self, n_pairs: int, rng: np.random.Generator) -> np.ndarray:
        if self.workspace is None:
            return rng.random(n_pairs) <= self.prob
        probs = self.workspace.request("blx_prob", (n_pairs,), np.float64)
        rng.random(out=probs)
        mask = self.workspace.request("blx_mask", (n_pairs,), np.bool_)
        np.less_equal(probs, self.prob, out=mask)
        return mask

    def _repair_random(self, values: np.ndarray, rng: np.random.Generator, key: str) -> None:
        mask_low = values < self.lower
        mask_high = values > self.upper
        mask = mask_low | mask_high
        if not np.any(mask):
            return
        span = self.upper - self.lower
        rand = self._rand(key, values.shape, rng)
        repaired = self.lower + rand * span
        values[mask] = repaired[mask]

    def _repair_reflect(self, values: np.ndarray) -> None:
        n_vars = self.lower.shape[0]
        for j in range(n_vars):
            low = self.lower[j]
            high = self.upper[j]
            width = high - low
            if width <= 0.0:
                values[:, j] = low
                continue
            val = values[:, j] - low
            period = 2.0 * width
            val = np.mod(val, period)
            over = val > width
            val[over] = period - val[over]
            values[:, j] = val + low

    def _repair_round(self, values: np.ndarray) -> None:
        np.rint(values, out=values)
        np.clip(values, self.lower, self.upper, out=values)

    def _apply_repair(self, child: np.ndarray, rng: np.random.Generator, key: str) -> None:
        if self.repair == "clip":
            np.clip(child, self.lower, self.upper, out=child)
        elif self.repair == "resample":
            self._repair_random(child, rng, key)
        elif self.repair == "reflect":
            self._repair_reflect(child)
        elif self.repair == "round":
            self._repair_round(child)
        else:
            raise ValueError(f"Unsupported BLX repair strategy '{self.repair}'.")

    def __call__(self, parents: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        parents_arr = self._as_matings(parents, copy=False, name="parents")
        offspring = parents_arr if self.allow_inplace else parents_arr.copy()
        self._check_bounds_match(offspring[:, 0, :], self.lower)
        n_pairs = offspring.shape[0]
        if n_pairs == 0:
            return offspring

        mask = self._sample_mask(n_pairs, rng)
        if not np.any(mask):
            return offspring

        active = offspring[mask]
        p1 = active[:, 0, :]
        p2 = active[:, 1, :]
        lo = np.minimum(p1, p2)
        hi = np.maximum(p1, p2)
        span = hi - lo
        lower = lo - self.alpha * span
        upper = hi + self.alpha * span
        width = upper - lower
        child1 = lower + self._rand("blx_rand1", lower.shape, rng) * width
        child2 = lower + self._rand("blx_rand2", lower.shape, rng) * width

        self._apply_repair(child1, rng, "blx_rand_repair1")
        self._apply_repair(child2, rng, "blx_rand_repair2")

        active[:, 0, :] = child1
        active[:, 1, :] = child2
        offspring[mask] = active
        return offspring


class ArithmeticCrossover(Crossover):
    """Arithmetic crossover mixing parents through random convex combinations."""

    def __init__(self, prob_crossover: float = 0.9) -> None:
        self.prob = float(prob_crossover)

    def __call__(self, parents: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        parents_arr = self._as_matings(parents, copy=False, name="parents")
        offspring = parents_arr.copy()
        n_pairs = offspring.shape[0]
        if n_pairs == 0:
            return offspring

        mask = np.asarray(rng.random(n_pairs) <= self.prob, dtype=bool)
        if not np.any(mask):
            return offspring

        lam = rng.random((int(mask.sum()), 1))
        p1 = offspring[mask, 0, :]
        p2 = offspring[mask, 1, :]
        child1 = lam * p1 + (1.0 - lam) * p2
        child2 = (1.0 - lam) * p1 + lam * p2
        offspring[mask, 0, :] = child1
        offspring[mask, 1, :] = child2
        return offspring


class DifferentialCrossover(Crossover):
    """Differential Evolution-style crossover/mutation operator."""

    def __init__(
        self,
        F: float = 0.5,
        CR: float = 0.9,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
    ) -> None:
        self.F = float(F)
        self.CR = float(CR)
        self.lower, self.upper = _ensure_bounds(lower, upper)

    def __call__(self, population: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        """Generate Differential Evolution trial vectors for the given population."""
        pop = self._as_population(population, name="population", copy=False)
        n_ind, n_vars = pop.shape
        _check_nvars(n_vars, self.lower)
        if n_ind < 3:
            raise ValueError("Differential crossover requires at least 3 individuals.")

        trial = pop.copy()
        all_indices = np.arange(n_ind)
        for i in range(n_ind):
            choices = np.delete(all_indices, i)
            r1, r2 = rng.choice(choices, size=2, replace=False)
            mutant = pop[i] + self.F * (pop[r1] - pop[r2])

            cross_mask = np.asarray(rng.random(n_vars) < self.CR, dtype=bool)
            j_rand = rng.integers(n_vars)
            cross_mask[j_rand] = True
            trial[i, cross_mask] = mutant[cross_mask]

        return _clip_population(trial, self.lower, self.upper)


class PCXCrossover(Crossover):
    """Parent-Centric Crossover (PCX) using 3-parent groups."""

    def __init__(
        self,
        sigma_eta: float = 0.1,
        sigma_zeta: float = 0.1,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
    ) -> None:
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
            if diff.shape[0] > 0:
                # Orthonormal basis spanning parent directions (shape n_vars x k)
                q, _ = np.linalg.qr(diff.T, mode="reduced")
                basis = q  # shape (n_vars, k)
            else:
                basis = np.eye(n_vars)
            avg_dist = np.mean(np.linalg.norm(diff, axis=1)) if diff.size else 0.0
            for j in range(3):
                noise = rng.normal(0.0, self.sigma_zeta * (avg_dist or 1.0), size=basis.shape[1])
                z = basis @ noise
                child = x0 + self.sigma_eta * d + z
                offspring[i, j, :] = np.clip(child, self.lower, self.upper)
        return offspring


class UNDXCrossover(Crossover):
    """Unimodal Normal Distribution Crossover (UNDX) aligned with jMetalPy."""

    def __init__(
        self,
        prob_crossover: float = 0.9,
        zeta: float = 0.5,
        eta: float = 0.35,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
    ) -> None:
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
                value1 = center[j] + alpha * diff[j] / distance + beta * orthogonal
                value2 = center[j] - alpha * diff[j] / distance - beta * orthogonal
                child1[j] = np.clip(value1, self.lower[j], self.upper[j])
                child2[j] = np.clip(value2, self.lower[j], self.upper[j])

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
    ) -> None:
        self.epsilon = float(epsilon)
        self.lower, self.upper = _ensure_bounds(lower, upper)

    def __call__(self, parents: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        groups = self._as_matings(parents, expected_parents=3, copy=False)
        n_groups, k, _ = groups.shape
        offspring = np.empty_like(groups)
        for i in range(n_groups):
            g = groups[i]
            centroid = np.mean(g, axis=0)
            for j in range(k):
                weights = np.asarray(rng.random(k), dtype=float)
                weights /= float(weights.sum())
                point = np.sum(weights[:, None] * g, axis=0)
                child = centroid + self.epsilon * (point - centroid)
                offspring[i, j, :] = np.clip(child, self.lower, self.upper)
        return offspring


__all__ = [
    "ArithmeticCrossover",
    "BLXAlphaCrossover",
    "Crossover",
    "DifferentialCrossover",
    "PCXCrossover",
    "SPXCrossover",
    "SBXCrossover",
    "UNDXCrossover",
]
