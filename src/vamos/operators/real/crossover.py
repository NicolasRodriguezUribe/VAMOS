"""Real-valued crossover operators."""

from __future__ import annotations

from abc import ABC, abstractmethod

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
        *,
        lower: ArrayLike,
        upper: ArrayLike,
        workspace: VariationWorkspace | None = None,
        allow_inplace: bool = False,
    ) -> None:
        self.prob = float(prob_crossover)
        self.eta = float(eta)
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

    def _buffer(self, key: str, shape: tuple[int, ...], dtype) -> np.ndarray:
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
        eps = 1.0e-14

        y1 = np.minimum(parent1, parent2)
        y2 = np.maximum(parent1, parent2)
        diff = y2 - y1
        valid = diff > eps
        if not np.any(valid):
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
        child1 = np.where(swap_mask, c2, c1)
        child2 = np.where(swap_mask, c1, c2)

        offspring[idx, 0, :] = child1
        offspring[idx, 1, :] = child2
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
        if normalized not in {"clip", "random"}:
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

        if self.repair == "clip":
            np.clip(child1, self.lower, self.upper, out=child1)
            np.clip(child2, self.lower, self.upper, out=child2)
        else:
            self._repair_random(child1, rng, "blx_rand_repair1")
            self._repair_random(child2, rng, "blx_rand_repair2")

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

        mask = rng.random(n_pairs) <= self.prob
        if not np.any(mask):
            return offspring

        lam = rng.random((mask.sum(), offspring.shape[2]))
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
        if n_ind < 4:
            raise ValueError("Differential crossover requires at least 4 individuals.")

        trial = pop.copy()
        all_indices = np.arange(n_ind)
        for i in range(n_ind):
            choices = np.delete(all_indices, i)
            r1, r2, r3 = rng.choice(choices, size=3, replace=False)
            mutant = pop[r1] + self.F * (pop[r2] - pop[r3])

            cross_mask = rng.random(n_vars) < self.CR
            j_rand = rng.integers(n_vars)
            cross_mask[j_rand] = True
            trial[i, cross_mask] = mutant[cross_mask]

        return _clip_population(trial, self.lower, self.upper)


__all__ = [
    "ArithmeticCrossover",
    "BLXAlphaCrossover",
    "Crossover",
    "DifferentialCrossover",
    "SBXCrossover",
]
