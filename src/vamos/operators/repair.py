from __future__ import annotations

from typing import Callable

import numpy as np


class GradientRepair:
    """
    Simple gradient-based repair: takes a step against constraint violation gradients.
    Assumes constraint_grad(X) returns shape (N, n_constraints, n_var).
    """

    def __init__(
        self,
        constraint_fun: Callable[[np.ndarray], np.ndarray],
        constraint_grad: Callable[[np.ndarray], np.ndarray],
        step_size: float = 0.1,
        max_iters: int = 3,
    ) -> None:
        self.constraint_fun = constraint_fun
        self.constraint_grad = constraint_grad
        self.step_size = float(step_size)
        self.max_iters = int(max_iters)

    def __call__(
        self,
        X: np.ndarray,
        xl: np.ndarray,
        xu: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        X_new = np.array(X, copy=True)
        for _ in range(self.max_iters):
            violations = self.constraint_fun(X_new)  # (N, m)
            grads = self.constraint_grad(X_new)  # (N, m, n_var)
            total_violation = violations.sum(axis=1, keepdims=True)
            if np.all(total_violation <= 1e-12):
                break
            # Aggregate gradients weighted by violation magnitude
            direction = (violations[..., None] * grads).sum(axis=1)
            X_new = X_new - self.step_size * direction
            X_new = np.clip(X_new, xl, xu)
        return X_new


__all__ = ["GradientRepair"]
