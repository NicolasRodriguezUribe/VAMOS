"""Real-valued mutation operators."""

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

    def _buffer(self, key: str, shape: tuple[int, ...], dtype) -> np.ndarray:
        if self.workspace is None:
            return np.empty(shape, dtype=dtype)
        return self.workspace.request(key, shape, dtype)

    def __call__(self, offspring: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        X = self._as_population(offspring, name="offspring", copy=False)
        n_ind, n_var = X.shape
        if n_ind == 0:
            return X
        self._check_bounds_match(X, self.lower)
        mask = self._mask((n_ind, n_var), rng)
        if not np.any(mask):
            return X

        yl = self.lower
        yu = self.upper
        rows, cols = np.nonzero(mask)
        if rows.size == 0:
            return X

        values = self._buffer("pm_values", (rows.size,), X.dtype)
        np.copyto(values, X[rows, cols])
        span_vals = self.span[cols]
        span_safe = self._span_safe[cols]
        delta1 = (values - yl[cols]) / span_safe
        delta2 = (yu[cols] - values) / span_safe
        rnd = self._rand("pm_rand", (rows.size,), rng)
        mut_pow = 1.0 / (self.eta + 1.0)
        deltaq = self._buffer("pm_deltaq", (rows.size,), np.float64)
        deltaq.fill(0.0)

        idx_lower = rnd <= 0.5
        idx_upper = ~idx_lower
        if np.any(idx_lower):
            xy = 1.0 - delta1[idx_lower]
            val = 2.0 * rnd[idx_lower] + (1.0 - 2.0 * rnd[idx_lower]) * np.power(xy, self.eta + 1.0)
            deltaq[idx_lower] = np.power(val, mut_pow) - 1.0
        if np.any(idx_upper):
            xy = 1.0 - delta2[idx_upper]
            val = 2.0 * (1.0 - rnd[idx_upper]) + 2.0 * (rnd[idx_upper] - 0.5) * np.power(xy, self.eta + 1.0)
            deltaq[idx_upper] = 1.0 - np.power(val, mut_pow)

        np.multiply(deltaq, span_vals, out=deltaq, casting="unsafe")
        np.add(values, deltaq, out=values, casting="unsafe")
        np.clip(values, yl[cols], yu[cols], out=values)
        X[rows, cols] = values
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
        delta = (
            self.workspace.request("nu_delta", X.shape, np.float64)
            if self.workspace
            else np.empty(X.shape, dtype=float)
        )
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


__all__ = [
    "GaussianMutation",
    "Mutation",
    "NonUniformMutation",
    "PolynomialMutation",
    "UniformResetMutation",
]
