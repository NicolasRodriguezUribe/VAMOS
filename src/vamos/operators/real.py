"""Real-coded evolutionary operators for crossover, mutation, and repair.

This module provides a small toolbox of variation operators used in
multi-objective evolutionary algorithms such as NSGA-II or MOEA/D.
All implementations rely solely on NumPy and operate on vectorized
arrays for clarity and efficiency.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

ArrayLike = np.ndarray


def _ensure_bounds(lower: ArrayLike, upper: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """Validate bounds and return float arrays of identical shape."""
    lower_arr = np.asarray(lower, dtype=float)
    upper_arr = np.asarray(upper, dtype=float)
    if lower_arr.shape != upper_arr.shape:
        raise ValueError("lower and upper bounds must have the same shape.")
    if lower_arr.ndim != 1:
        raise ValueError("Bounds must be one-dimensional arrays.")
    if np.any(lower_arr > upper_arr):
        raise ValueError("Each lower bound must be <= corresponding upper bound.")
    return lower_arr, upper_arr


def _clip_population(x: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> np.ndarray:
    """Return a clipped copy of x inside [lower, upper]."""
    return np.clip(x, lower, upper)


def _check_nvars(n_vars: int, bounds: np.ndarray) -> None:
    """Ensure that a bounds array matches the provided dimensionality."""
    if bounds.shape[0] != n_vars:
        raise ValueError("Bounds dimensionality does not match the individual size.")


class VariationWorkspace:
    """
    Simple buffer registry that hands out reusable NumPy arrays keyed by name.
    Operators can reuse temporary arrays across generations to avoid reallocations.
    """

    def __init__(self):
        self._buffers: dict[str, np.ndarray] = {}

    def request(self, key: str, shape: tuple[int, ...], dtype) -> np.ndarray:
        dtype = np.dtype(dtype)
        buf = self._buffers.get(key)
        if buf is None or buf.shape != shape or buf.dtype != dtype:
            buf = np.empty(shape, dtype=dtype)
            self._buffers[key] = buf
        return buf


class Crossover(ABC):
    """Base class for real-coded crossover operators."""

    @abstractmethod
    def __call__(
        self,
        parents: ArrayLike,
        rng: np.random.Generator,
    ) -> ArrayLike:
        """
        Apply crossover to a set of parents.

        Parameters
        ----------
        parents : np.ndarray
            Array of parents with shape (n_matings, 2, n_vars).
            Each row contains two parents to be mated.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        offspring : np.ndarray
            Offspring array with shape (n_matings, 2, n_vars).
        """
        raise NotImplementedError


class Mutation(ABC):
    """Base class for real-coded mutation operators."""

    @abstractmethod
    def __call__(
        self,
        offspring: ArrayLike,
        rng: np.random.Generator,
    ) -> ArrayLike:
        """
        Mutate a batch of individuals in-place or return a mutated copy.

        Parameters
        ----------
        offspring : np.ndarray
            Array of individuals with shape (n_individuals, n_vars).
        rng : np.random.Generator

        Returns
        -------
        mutated : np.ndarray
            Mutated individuals with shape (n_individuals, n_vars).
        """
        raise NotImplementedError


class Repair(ABC):
    """Base class for repair operators (bounds and feasibility helpers)."""

    @abstractmethod
    def __call__(
        self,
        x: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
        rng: np.random.Generator,
    ) -> ArrayLike:
        """
        Repair a batch of individuals so that they respect variable bounds.

        Parameters
        ----------
        x : np.ndarray
            Array of individuals, shape (n_individuals, n_vars).
        lower : np.ndarray
            Lower bounds, shape (n_vars,).
        upper : np.ndarray
            Upper bounds, shape (n_vars,).
        rng : np.random.Generator

        Returns
        -------
        repaired : np.ndarray
            Repaired individuals, shape (n_individuals, n_vars).
        """
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
    ) -> None:
        self.prob = float(prob_crossover)
        self.eta = float(eta)
        self.lower, self.upper = _ensure_bounds(lower, upper)

    def __call__(self, parents: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        parents = np.asarray(parents, dtype=float)
        if parents.ndim != 3 or parents.shape[1] != 2:
            raise ValueError("parents must have shape (n_matings, 2, n_vars).")
        offspring = parents.copy()
        n_pairs, _, n_vars = offspring.shape
        _check_nvars(n_vars, self.lower)
        if n_pairs == 0:
            return offspring

        apply_mask = rng.random(n_pairs) <= self.prob
        if not np.any(apply_mask):
            return offspring

        active = offspring[apply_mask]
        parent1 = active[:, 0, :].copy()
        parent2 = active[:, 1, :].copy()
        eps = 1.0e-14

        y1 = np.minimum(parent1, parent2)
        y2 = np.maximum(parent1, parent2)
        diff = y2 - y1
        valid = diff > eps
        if not np.any(valid):
            return offspring

        xl = self.lower.reshape(1, -1)
        xu = self.upper.reshape(1, -1)
        rand = rng.random(parent1.shape)
        beta = np.empty_like(parent1)
        betaq = np.empty_like(parent1)

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

        c1 = np.clip(c1, self.lower, self.upper)
        c2 = np.clip(c2, self.lower, self.upper)
        swap = rng.random(parent1.shape) <= 0.5
        child1 = np.where(swap, c2, c1)
        child2 = np.where(swap, c1, c2)

        active[:, 0, :] = child1
        active[:, 1, :] = child2
        offspring[apply_mask] = active
        return offspring


class BLXAlphaCrossover(Crossover):
    """Blend crossover (BLX-α) with optional buffer reuse and repair strategies."""

    def __init__(
        self,
        alpha: float = 0.5,
        prob_crossover: float = 0.9,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
        repair: str = "clip",
        workspace: VariationWorkspace | None = None,
    ) -> None:
        self.alpha = float(alpha)
        self.prob = float(prob_crossover)
        self.lower, self.upper = _ensure_bounds(lower, upper)
        normalized = (repair or "clip").lower()
        if normalized not in {"clip", "random"}:
            raise ValueError(f"Unsupported BLX repair strategy '{repair}'.")
        self.repair = normalized
        self.workspace = workspace

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
        parents = np.asarray(parents, dtype=float)
        if parents.ndim != 3 or parents.shape[1] != 2:
            raise ValueError("parents must have shape (n_matings, 2, n_vars).")
        offspring = parents.copy()
        _check_nvars(parents.shape[2], self.lower)
        n_pairs = parents.shape[0]
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
        parents = np.asarray(parents, dtype=float)
        if parents.ndim != 3 or parents.shape[1] != 2:
            raise ValueError("parents must have shape (n_matings, 2, n_vars).")
        offspring = parents.copy()
        n_pairs = parents.shape[0]
        if n_pairs == 0:
            return offspring

        mask = rng.random(n_pairs) <= self.prob
        if not np.any(mask):
            return offspring

        lam = rng.random((mask.sum(), parents.shape[2]))
        p1 = parents[mask, 0, :]
        p2 = parents[mask, 1, :]
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
        pop = np.asarray(population, dtype=float)
        if pop.ndim != 2:
            raise ValueError("population must have shape (n_individuals, n_vars).")
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


class PolynomialMutation(Mutation):
    """Standard polynomial mutation used in NSGA-II."""

    def __init__(
        self,
        prob_mutation: float,
        eta: float = 20.0,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
    ) -> None:
        self.prob = float(prob_mutation)
        self.eta = float(eta)
        self.lower, self.upper = _ensure_bounds(lower, upper)

    def __call__(self, offspring: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        X = np.asarray(offspring, dtype=float).copy()
        if X.ndim != 2:
            raise ValueError("offspring must have shape (n_individuals, n_vars).")
        n_ind, n_var = X.shape
        _check_nvars(n_var, self.lower)
        if n_ind == 0:
            return X
        mask = rng.random((n_ind, n_var)) <= self.prob
        if not np.any(mask):
            return X

        yl = self.lower
        yu = self.upper
        eta = self.eta
        span = yu - yl
        span_safe = np.where(span == 0.0, 1.0, span)
        delta1 = (X - yl) / span_safe
        delta2 = (yu - X) / span_safe
        rnd = rng.random((n_ind, n_var))
        mut_pow = 1.0 / (eta + 1.0)
        deltaq = np.zeros_like(X)
        idx_lower = mask & (rnd <= 0.5)
        idx_upper = mask & ~idx_lower
        if np.any(idx_lower):
            xy = 1.0 - delta1[idx_lower]
            val = 2.0 * rnd[idx_lower] + (1.0 - 2.0 * rnd[idx_lower]) * np.power(xy, eta + 1.0)
            deltaq[idx_lower] = np.power(val, mut_pow) - 1.0
        if np.any(idx_upper):
            xy = 1.0 - delta2[idx_upper]
            val = 2.0 * (1.0 - rnd[idx_upper]) + 2.0 * (rnd[idx_upper] - 0.5) * np.power(xy, eta + 1.0)
            deltaq[idx_upper] = 1.0 - np.power(val, mut_pow)

        X += deltaq * span
        return _clip_population(X, yl, yu)


class GaussianMutation(Mutation):
    """Gaussian mutation with optional bounds clamping."""

    def __init__(
        self,
        prob_mutation: float,
        sigma: float | ArrayLike,
        *,
        lower: ArrayLike | None = None,
        upper: ArrayLike | None = None,
    ) -> None:
        self.prob = float(prob_mutation)
        sigma_arr = np.asarray(sigma, dtype=float)
        if sigma_arr.ndim == 0:
            self.sigma = sigma_arr
        elif sigma_arr.ndim == 1:
            self.sigma = sigma_arr
        else:
            raise ValueError("sigma must be scalar or 1-D array.")
        if lower is None or upper is None:
            self.lower = self.upper = None
        else:
            self.lower, self.upper = _ensure_bounds(lower, upper)

    def __call__(self, offspring: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        X = np.asarray(offspring, dtype=float).copy()
        if X.ndim != 2:
            raise ValueError("offspring must have shape (n_individuals, n_vars).")
        n_ind, n_var = X.shape
        if self.lower is not None and self.upper is not None:
            _check_nvars(n_var, self.lower)
        if n_ind == 0:
            return X
        mask = rng.random((n_ind, n_var)) <= self.prob
        if not np.any(mask):
            return X
        if np.ndim(self.sigma) == 0:
            noise = rng.normal(0.0, float(self.sigma), size=X.shape)
        else:
            sigma = np.broadcast_to(self.sigma, (n_ind, n_var))
            noise = rng.normal(0.0, sigma)
        X[mask] += noise[mask]
        if self.lower is not None and self.upper is not None:
            X = _clip_population(X, self.lower, self.upper)
        return X


class UniformResetMutation(Mutation):
    """Uniform reset mutation that resamples genes inside their bounds."""

    def __init__(
        self,
        prob_mutation: float,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
    ) -> None:
        self.prob = float(prob_mutation)
        self.lower, self.upper = _ensure_bounds(lower, upper)

    def __call__(self, offspring: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        X = np.asarray(offspring, dtype=float).copy()
        if X.ndim != 2:
            raise ValueError("offspring must have shape (n_individuals, n_vars).")
        _check_nvars(X.shape[1], self.lower)
        mask = rng.random(X.shape) <= self.prob
        if not np.any(mask):
            return X
        lower = np.broadcast_to(self.lower, X.shape)
        upper = np.broadcast_to(self.upper, X.shape)
        resampled = lower + rng.random(X.shape) * (upper - lower)
        X[mask] = resampled[mask]
        return X


class NonUniformMutation(Mutation):
    """
    Non-uniform mutation that samples perturbations biased toward the bounds
    using the simplified formulation employed by our NSGA-II variant.
    """

    def __init__(
        self,
        prob_mutation: float,
        perturbation: float = 0.5,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
        workspace: VariationWorkspace | None = None,
    ) -> None:
        self.prob = float(np.clip(prob_mutation, 0.0, 1.0))
        self.perturbation = max(float(perturbation), 1e-8)
        self.lower, self.upper = _ensure_bounds(lower, upper)
        self.span = self.upper - self.lower
        self.workspace = workspace

    def _rand(self, key: str, shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
        if self.workspace is None:
            return rng.random(shape)
        buf = self.workspace.request(key, shape, np.float64)
        rng.random(out=buf)
        return buf

    def _mask(self, shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
        if self.workspace is None:
            return rng.random(shape) <= self.prob
        probs = self.workspace.request("nu_prob", shape, np.float64)
        rng.random(out=probs)
        mask = self.workspace.request("nu_mask", shape, np.bool_)
        np.less_equal(probs, self.prob, out=mask)
        return mask

    def __call__(self, offspring: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        X = np.asarray(offspring, dtype=float).copy()
        if X.ndim != 2:
            raise ValueError("offspring must have shape (n_individuals, n_vars).")
        _check_nvars(X.shape[1], self.lower)
        mask = self._mask(X.shape, rng)
        if not np.any(mask):
            return X

        rand = self._rand("nu_rand", X.shape, rng)
        np.power(rand, self.perturbation, out=rand)
        delta = self.workspace.request("nu_delta", X.shape, np.float64) if self.workspace else np.empty(X.shape, dtype=float)
        np.subtract(1.0, rand, out=delta)
        delta *= self.span
        direction_rand = self._rand("nu_direction", X.shape, rng)
        direction = np.where(direction_rand <= 0.5, -1.0, 1.0)
        update = delta * direction
        if self.workspace is None:
            X += np.where(mask, update, 0.0)
        else:
            buffered_update = self.workspace.request("nu_update", X.shape, np.float64)
            buffered_update.fill(0.0)
            buffered_update[mask] = update[mask]
            X += buffered_update
        np.clip(X, self.lower, self.upper, out=X)
        return X


class ClampRepair(Repair):
    """Repair operator that clamps variables to their bounds."""

    def __call__(
        self,
        x: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
        rng: np.random.Generator,  # pylint: disable=unused-argument
    ) -> ArrayLike:
        lower_arr, upper_arr = _ensure_bounds(lower, upper)
        return _clip_population(np.asarray(x, dtype=float), lower_arr, upper_arr)


class ReflectRepair(Repair):
    """Repair operator that reflects out-of-bounds values back into range."""

    def __call__(
        self,
        x: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
        rng: np.random.Generator,  # pylint: disable=unused-argument
    ) -> ArrayLike:
        lower_arr, upper_arr = _ensure_bounds(lower, upper)
        result = np.asarray(x, dtype=float).copy()
        n_vars = lower_arr.shape[0]
        for j in range(n_vars):
            low = lower_arr[j]
            high = upper_arr[j]
            width = high - low
            if width <= 0.0:
                result[:, j] = low
                continue
            val = result[:, j] - low
            period = 2.0 * width
            val = np.mod(val, period)
            over = val > width
            val[over] = period - val[over]
            result[:, j] = val + low
        return result


class ResampleRepair(Repair):
    """Repair operator that resamples violated genes uniformly inside bounds."""

    def __call__(
        self,
        x: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
        rng: np.random.Generator,
    ) -> ArrayLike:
        lower_arr, upper_arr = _ensure_bounds(lower, upper)
        result = np.asarray(x, dtype=float).copy()
        mask_low = result < lower_arr
        mask_high = result > upper_arr
        mask = mask_low | mask_high
        if not np.any(mask):
            return result
        span = upper_arr - lower_arr
        resampled = lower_arr + rng.random(result.shape) * span
        result[mask] = resampled[mask]
        return result


class RoundRepair(Repair):
    """Repair operator that rounds genes to the nearest integer inside the bounds."""

    def __call__(
        self,
        x: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
        rng: np.random.Generator,  # pylint: disable=unused-argument
    ) -> ArrayLike:
        lower_arr, upper_arr = _ensure_bounds(lower, upper)
        rounded = np.rint(np.asarray(x, dtype=float))
        return _clip_population(rounded, lower_arr, upper_arr)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n_ind, n_vars = 5, 3
    lower = np.array([-5.0, 0.0, 1.0])
    upper = np.array([5.0, 10.0, 3.0])

    pop = rng.uniform(lower, upper, size=(n_ind, n_vars))

    parents = np.stack(
        [pop[rng.integers(0, n_ind, size=2)] for _ in range(4)],
        axis=0,
    )

    sbx = SBXCrossover(prob_crossover=0.9, eta=10.0, lower=lower, upper=upper)
    blx = BLXAlphaCrossover(alpha=0.5, prob_crossover=0.9, lower=lower, upper=upper)
    poly_mut = PolynomialMutation(prob_mutation=1.0 / n_vars, eta=20.0, lower=lower, upper=upper)
    clamp = ClampRepair()

    off_sbx = sbx(parents, rng)
    off_blx = blx(parents, rng)
    mut_pop = poly_mut(pop.copy(), rng)
    repaired = clamp(mut_pop.copy(), lower, upper, rng)

    print("Parents:\n", parents)
    print("SBX offspring:\n", off_sbx)
    print("BLX offspring:\n", off_blx)
    print("Mutated population:\n", mut_pop)
    print("Repaired population:\n", repaired)
