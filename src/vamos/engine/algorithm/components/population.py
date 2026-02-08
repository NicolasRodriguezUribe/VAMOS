from __future__ import annotations

from typing import Any

import numpy as np

from vamos.foundation.encoding import EncodingLike, normalize_encoding
from vamos.operators.impl.binary import random_binary_population
from vamos.operators.impl.integer import random_integer_population
from vamos.operators.impl.mixed import mixed_initialize
from vamos.operators.impl.permutation import random_permutation_population
from vamos.operators.impl.real import LatinHypercubeInitializer, ScatterSearchInitializer


def resolve_bounds(problem: Any, encoding: EncodingLike) -> tuple[np.ndarray, np.ndarray]:
    normalized = normalize_encoding(encoding)
    bounds_dtype = int if normalized == "integer" else float
    xl = np.asarray(problem.xl, dtype=bounds_dtype)
    xu = np.asarray(problem.xu, dtype=bounds_dtype)
    n_var = problem.n_var
    if xl.ndim == 0:
        xl = np.full(n_var, xl, dtype=bounds_dtype)
    if xu.ndim == 0:
        xu = np.full(n_var, xu, dtype=bounds_dtype)
    xl = np.ascontiguousarray(xl, dtype=bounds_dtype)
    xu = np.ascontiguousarray(xu, dtype=bounds_dtype)
    return xl, xu


def initialize_population(
    pop_size: int,
    n_var: int,
    xl: np.ndarray,
    xu: np.ndarray,
    encoding: EncodingLike,
    rng: np.random.Generator,
    problem: Any | None = None,
    initializer: Any | None = None,
) -> np.ndarray:
    normalized = normalize_encoding(encoding)
    if pop_size <= 0:
        raise ValueError("pop_size must be positive.")
    if initializer is not None and callable(initializer):
        return np.asarray(initializer(), dtype=float)
    if initializer is not None and isinstance(initializer, dict):
        init_type = initializer.get("type", "random").lower()
        if init_type == "custom":
            custom_fn = initializer.get("fn")
            if callable(custom_fn):
                return np.asarray(custom_fn(), dtype=float)
            raise ValueError("Custom initializer requires a callable 'fn'.")
        if init_type == "lhs":
            return LatinHypercubeInitializer(pop_size, xl, xu, rng=rng)()
        if init_type in {"scatter", "scatter_search"}:
            base = int(initializer.get("base_size", max(20, pop_size)))
            return ScatterSearchInitializer(pop_size, xl, xu, base_size=base, rng=rng)()
    if normalized == "permutation":
        return random_permutation_population(pop_size, n_var, rng)
    if normalized == "binary":
        return random_binary_population(pop_size, n_var, rng)
    if normalized == "integer":
        return random_integer_population(pop_size, n_var, xl.astype(int), xu.astype(int), rng)
    if normalized == "mixed":
        spec = getattr(problem, "mixed_spec", None)
        if spec is None:
            raise ValueError("Mixed-encoding problems must define 'mixed_spec'.")
        return mixed_initialize(pop_size, n_var, spec, rng)
    return rng.uniform(xl, xu, size=(pop_size, n_var))


def evaluate_population(problem: Any, X: np.ndarray) -> np.ndarray:
    out = {"F": np.empty((X.shape[0], problem.n_obj))}
    problem.evaluate(X, out)
    return out["F"]


__all__ = ["resolve_bounds", "initialize_population", "evaluate_population"]
