from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats.qmc import Halton, Sobol  # type: ignore[import-untyped]


class LatinHypercubeInitializer:
    """
    Latin Hypercube Sampling initializer for real-valued variables.
    Generates n_solutions points inside [lower, upper].
    """

    def __init__(
        self,
        n_solutions: int,
        lower: np.ndarray,
        upper: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.n_solutions = int(n_solutions)
        if self.n_solutions <= 0:
            raise ValueError("n_solutions must be positive.")
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        if self.lower.shape != self.upper.shape:
            raise ValueError("lower and upper must have the same shape.")
        if self.lower.ndim != 1:
            raise ValueError("Bounds must be one-dimensional.")
        self.n_var = self.lower.shape[0]
        self.rng = rng or np.random.default_rng()

    def __call__(self) -> np.ndarray:
        n = self.n_solutions
        d = self.n_var
        samples = np.empty((n, d), dtype=float)
        for j in range(d):
            strata = (np.arange(n, dtype=float) + self.rng.random(n)) / n
            self.rng.shuffle(strata)
            samples[:, j] = strata
        span = self.upper - self.lower
        return np.asarray(self.lower + samples * span, dtype=float)


class ScatterSearchInitializer:
    """
    Scatter search-style initializer using an LHS base set and convex combinations.
    """

    def __init__(
        self,
        n_solutions: int,
        lower: np.ndarray,
        upper: np.ndarray,
        base_size: int = 20,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.n_solutions = int(n_solutions)
        if self.n_solutions <= 0:
            raise ValueError("n_solutions must be positive.")
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        if self.lower.shape != self.upper.shape:
            raise ValueError("lower and upper must have the same shape.")
        if self.lower.ndim != 1:
            raise ValueError("Bounds must be one-dimensional.")
        self.base_size = max(1, int(base_size))
        self.rng = rng or np.random.default_rng()

    def __call__(self) -> np.ndarray:
        lhs = LatinHypercubeInitializer(self.base_size, self.lower, self.upper, rng=self.rng)
        base = lhs()
        if self.n_solutions <= self.base_size:
            return base[: self.n_solutions]
        needed = self.n_solutions - self.base_size
        children = np.empty((needed, self.lower.shape[0]), dtype=float)
        for i in range(needed):
            idx_a = self.rng.integers(0, self.base_size)
            idx_b = self.rng.integers(0, self.base_size)
            alpha = self.rng.random()
            child = alpha * base[idx_a] + (1.0 - alpha) * base[idx_b]
            np.clip(child, self.lower, self.upper, out=child)
            children[i] = child
        return np.vstack((base, children))[: self.n_solutions]


class SobolInitializer:
    """Sobol quasi-random sequence initializer for real-valued variables."""

    def __init__(
        self,
        n_solutions: int,
        lower: np.ndarray,
        upper: np.ndarray,
        rng: np.random.Generator | None = None,
        scramble: bool = True,
    ) -> None:
        self.n_solutions = int(n_solutions)
        if self.n_solutions <= 0:
            raise ValueError("n_solutions must be positive.")
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        if self.lower.shape != self.upper.shape:
            raise ValueError("lower and upper must have the same shape.")
        if self.lower.ndim != 1:
            raise ValueError("Bounds must be one-dimensional.")
        self.n_var = self.lower.shape[0]
        self.scramble = scramble
        self.rng = rng or np.random.default_rng()

    def __call__(self) -> np.ndarray:
        seed = int(self.rng.integers(0, 2**31))
        sampler = Sobol(d=self.n_var, scramble=self.scramble, seed=seed)
        m = int(np.ceil(np.log2(max(self.n_solutions, 2))))
        n_pow2 = 2**m
        if n_pow2 < self.n_solutions:
            n_pow2 *= 2
        samples = sampler.random(n_pow2)[: self.n_solutions]
        span = self.upper - self.lower
        return np.asarray(self.lower + samples * span, dtype=float)


class HaltonInitializer:
    """Halton quasi-random sequence initializer for real-valued variables."""

    def __init__(
        self,
        n_solutions: int,
        lower: np.ndarray,
        upper: np.ndarray,
        rng: np.random.Generator | None = None,
        scramble: bool = True,
    ) -> None:
        self.n_solutions = int(n_solutions)
        if self.n_solutions <= 0:
            raise ValueError("n_solutions must be positive.")
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        if self.lower.shape != self.upper.shape:
            raise ValueError("lower and upper must have the same shape.")
        if self.lower.ndim != 1:
            raise ValueError("Bounds must be one-dimensional.")
        self.n_var = self.lower.shape[0]
        self.scramble = scramble
        self.rng = rng or np.random.default_rng()

    def __call__(self) -> np.ndarray:
        seed = int(self.rng.integers(0, 2**31))
        sampler = Halton(d=self.n_var, scramble=self.scramble, seed=seed)
        samples = sampler.random(self.n_solutions)
        span = self.upper - self.lower
        return np.asarray(self.lower + samples * span, dtype=float)


class OppositionBasedInitializer:
    """Opposition-Based Learning (OBL) initializer for real-valued variables.

    Generates a random population and its opposition-based counterparts
    (opposite = lower + upper - x), evaluates all 2*n_solutions candidates,
    and selects the best n_solutions.
    """

    def __init__(
        self,
        n_solutions: int,
        lower: np.ndarray,
        upper: np.ndarray,
        problem: Any,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.n_solutions = int(n_solutions)
        if self.n_solutions <= 0:
            raise ValueError("n_solutions must be positive.")
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        if self.lower.shape != self.upper.shape:
            raise ValueError("lower and upper must have the same shape.")
        if self.lower.ndim != 1:
            raise ValueError("Bounds must be one-dimensional.")
        self.n_var = self.lower.shape[0]
        self.problem = problem
        self.rng = rng or np.random.default_rng()

    def __call__(self) -> np.ndarray:
        X = self.rng.uniform(self.lower, self.upper, size=(self.n_solutions, self.n_var))
        opposites = self.lower + self.upper - X
        np.clip(opposites, self.lower, self.upper, out=opposites)
        combined = np.vstack((X, opposites))

        out: dict[str, Any] = {"F": np.empty((combined.shape[0], self.problem.n_obj))}
        self.problem.evaluate(combined, out)
        F = out["F"]

        indices = self._select_best(F)
        return np.asarray(combined[indices], dtype=float)

    def _select_best(self, F: np.ndarray) -> np.ndarray:
        n = self.n_solutions
        if F.shape[1] == 1:
            return np.argsort(F[:, 0])[:n]
        from vamos.foundation.kernel.numpy_backend import (
            _compute_crowding,
            _fast_non_dominated_sort,
            _select_nsga2,
        )

        fronts, _rank = _fast_non_dominated_sort(F)
        crowding = _compute_crowding(F, fronts)
        return _select_nsga2(fronts, crowding, n)


__all__ = [
    "LatinHypercubeInitializer",
    "ScatterSearchInitializer",
    "SobolInitializer",
    "HaltonInitializer",
    "OppositionBasedInitializer",
]
