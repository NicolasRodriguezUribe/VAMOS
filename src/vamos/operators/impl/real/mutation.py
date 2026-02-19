"""Real-valued mutation operators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .repair import Repair
from .utils import (
    ArrayLike,
    RealOperator,
    VariationWorkspace,
    _check_nvars,
    _clip_population,
    _ensure_bounds,
)


class Mutation(RealOperator, ABC):
    """Base class for real-coded mutation operators."""

    @abstractmethod
    def __call__(self, offspring: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        raise NotImplementedError


class PolynomialMutation(Mutation):
    """Standard polynomial mutation used in NSGA-II."""

    def __init__(
        self,
        prob_mutation: float,
        eta: float = 20.0,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
        workspace: VariationWorkspace | None = None,
    ) -> None:
        self.prob = float(prob_mutation)
        self.eta = float(eta)
        self.lower, self.upper = _ensure_bounds(lower, upper)
        self.span = self.upper - self.lower
        self._span_safe = np.where(self.span == 0.0, 1.0, self.span)
        self.workspace = workspace

    def _mask(self, shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
        if self.workspace is None:
            return rng.random(shape) <= self.prob
        probs = self.workspace.request("pm_prob", shape, np.float64)
        rng.random(out=probs)
        mask = self.workspace.request("pm_mask", shape, np.bool_)
        np.less_equal(probs, self.prob, out=mask)
        return mask

    def _rand(self, key: str, shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
        if self.workspace is None:
            return rng.random(shape)
        buf = self.workspace.request(key, shape, np.float64)
        rng.random(out=buf)
        return buf

    def _buffer(self, key: str, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
        if self.workspace is None:
            return np.empty(shape, dtype=dtype)
        return self.workspace.request(key, shape, dtype)

    def __call__(self, offspring: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        X = self._as_population(offspring, name="offspring")
        n_ind, n_var = X.shape
        if n_ind == 0:
            return X
        self._check_bounds_match(X, self.lower)

        # Draw full random grids to keep RNG consumption aligned with regression fixtures:
        # first grid for activation mask, second grid for delta sampling.
        rnd_mask = rng.random(X.shape)
        rnd_delta = rng.random(X.shape)
        mask = rnd_mask <= self.prob
        if not np.any(mask):
            return X

        mut_pow = 1.0 / (self.eta + 1.0)
        rows, cols = np.nonzero(mask)
        for i, j in zip(rows, cols):
            y = X[i, j]
            yl = self.lower[j]
            yu = self.upper[j]
            if yu <= yl:
                continue
            delta1 = (y - yl) / (yu - yl)
            delta2 = (yu - y) / (yu - yl)
            rnd = rnd_delta[i, j]
            if rnd <= 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (xy ** (self.eta + 1.0))
                deltaq = val**mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (xy ** (self.eta + 1.0))
                deltaq = 1.0 - val**mut_pow
            y += deltaq * (yu - yl)
            y = min(max(y, yl), yu)
            X[i, j] = y
        return X


class GaussianMutation(Mutation):
    """Gaussian mutation with optional bounds clamping."""

    def __init__(
        self,
        prob_mutation: float,
        sigma: float | ArrayLike,
        *,
        lower: ArrayLike | None = None,
        upper: ArrayLike | None = None,
        workspace: VariationWorkspace | None = None,
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
        X = self._as_population(offspring, name="offspring")
        n_ind, n_var = X.shape
        if n_ind == 0:
            return X
        if self.lower is not None and self.upper is not None:
            _check_nvars(n_var, self.lower)
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
        workspace: VariationWorkspace | None = None,
    ) -> None:
        self.prob = float(prob_mutation)
        self.lower, self.upper = _ensure_bounds(lower, upper)

    def __call__(self, offspring: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        X = self._as_population(offspring, name="offspring")
        self._check_bounds_match(X, self.lower)
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
        X = self._as_population(offspring, name="offspring")
        self._check_bounds_match(X, self.lower)
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


class UniformMutation(Mutation):
    """Uniform mutation with bounded perturbation scaled by variable range."""

    def __init__(
        self,
        prob_mutation: float | None = None,
        perturb: float = 0.1,
        *,
        prob: float | None = None,
        lower: ArrayLike,
        upper: ArrayLike,
        repair: Repair | None = None,
        rng: np.random.Generator | None = None,
        workspace: VariationWorkspace | None = None,
    ) -> None:
        if prob_mutation is None and prob is None:
            raise TypeError("UniformMutation requires 'prob' (legacy) or 'prob_mutation'.")
        if prob_mutation is not None and prob is not None:
            raise TypeError("UniformMutation received both 'prob' and 'prob_mutation'. Use only one.")
        prob_value = prob_mutation if prob is None else prob
        if prob_value is None:  # pragma: no cover - guarded by checks above
            raise TypeError("UniformMutation requires a mutation probability.")
        self.prob = float(prob_value)
        self.perturb = float(np.clip(perturb, 0.0, 1.0))
        self.lower, self.upper = _ensure_bounds(lower, upper)
        self.range = self.upper - self.lower
        self.repair = repair
        self.rng = rng

    def __call__(self, offspring: ArrayLike, rng: np.random.Generator | None = None) -> ArrayLike:
        arr = np.asarray(offspring, dtype=float)
        squeeze = False
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
            squeeze = True
        X = self._as_population(arr, name="offspring")
        n_ind, n_var = X.shape
        active_rng = rng or self.rng or np.random.default_rng()
        mask = active_rng.random((n_ind, n_var)) <= self.prob
        if not np.any(mask):
            return X[0] if squeeze else X
        delta = active_rng.uniform(-self.perturb, self.perturb, size=X.shape)
        update = delta * self.range
        X = X.copy()
        X += np.where(mask, update, 0.0)
        if self.repair is not None:
            X = self.repair(X, self.lower, self.upper, active_rng)
        else:
            X = _clip_population(X, self.lower, self.upper)
        return X[0] if squeeze else X


class LinkedPolynomialMutation(Mutation):
    """Polynomial mutation using a shared delta across all mutated variables."""

    def __init__(
        self,
        prob_mutation: float | None = None,
        eta: float = 20.0,
        *,
        prob: float | None = None,
        lower: ArrayLike,
        upper: ArrayLike,
        repair: Repair | None = None,
        rng: np.random.Generator | None = None,
        workspace: VariationWorkspace | None = None,
    ) -> None:
        if prob_mutation is None and prob is None:
            raise TypeError("LinkedPolynomialMutation requires 'prob' (legacy) or 'prob_mutation'.")
        if prob_mutation is not None and prob is not None:
            raise TypeError("LinkedPolynomialMutation received both 'prob' and 'prob_mutation'. Use only one.")
        prob_value = prob_mutation if prob is None else prob
        if prob_value is None:  # pragma: no cover - guarded by checks above
            raise TypeError("LinkedPolynomialMutation requires a mutation probability.")
        self.prob = float(prob_value)
        self.eta = float(eta)
        self.lower, self.upper = _ensure_bounds(lower, upper)
        self.span = self.upper - self.lower
        self.repair = repair
        self.rng = rng

    def __call__(self, offspring: ArrayLike, rng: np.random.Generator | None = None) -> ArrayLike:
        arr = np.asarray(offspring, dtype=float)
        squeeze = False
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
            squeeze = True
        X = self._as_population(arr, name="offspring")
        active_rng = rng or self.rng or np.random.default_rng()
        n_ind, n_var = X.shape
        mask = active_rng.random((n_ind, n_var)) <= self.prob
        if not np.any(mask):
            return X[0] if squeeze else X
        u = active_rng.random()
        if u < 0.5:
            delta = (2.0 * u) ** (1.0 / (self.eta + 1.0)) - 1.0
        else:
            delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (self.eta + 1.0))
        update = delta * self.span
        X = X.copy()
        X += np.where(mask, update, 0.0)
        if self.repair is not None:
            X = self.repair(X, self.lower, self.upper, active_rng)
        else:
            X = _clip_population(X, self.lower, self.upper)
        return X[0] if squeeze else X


class CauchyMutation(Mutation):
    """Cauchy mutation with optional bounds clamping."""

    def __init__(
        self,
        prob_mutation: float,
        gamma: float = 0.1,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
        workspace: VariationWorkspace | None = None,
    ) -> None:
        self.prob = float(prob_mutation)
        self.gamma = float(gamma)
        self.lower, self.upper = _ensure_bounds(lower, upper)

    def __call__(self, offspring: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        X = self._as_population(offspring, name="offspring")
        self._check_bounds_match(X, self.lower)
        mask = rng.random(X.shape) <= self.prob
        if not np.any(mask):
            return X
        noise = rng.standard_cauchy(size=X.shape) * self.gamma
        X[mask] += noise[mask]
        np.clip(X, self.lower, self.upper, out=X)
        return X


__all__ = [
    "CauchyMutation",
    "GaussianMutation",
    "LinkedPolynomialMutation",
    "Mutation",
    "NonUniformMutation",
    "PolynomialMutation",
    "UniformMutation",
    "UniformResetMutation",
]
