from __future__ import annotations

import numpy as np
from typing import Optional


class LatinHypercubeInitializer:
    """
    Latin Hypercube Sampling initializer for real-valued variables.
    Generates n_solutions points inside [lower, upper].
    """

    def __init__(self, n_solutions: int, lower: np.ndarray, upper: np.ndarray, rng: Optional[np.random.Generator] = None):
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
        return self.lower + samples * span


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
        rng: Optional[np.random.Generator] = None,
    ):
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


__all__ = ["LatinHypercubeInitializer", "ScatterSearchInitializer"]
