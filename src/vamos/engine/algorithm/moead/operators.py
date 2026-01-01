# algorithm/moead/operators.py
"""
Operator building for MOEA/D.

This module handles the construction of variation operators (crossover and mutation)
for different encodings (continuous, binary, integer).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from vamos.engine.algorithm.components.utils import resolve_prob_expression
from vamos.operators.binary import (
    bit_flip_mutation,
    one_point_crossover,
    two_point_crossover,
    uniform_crossover,
)
from vamos.operators.integer import (
    arithmetic_integer_crossover,
    creep_mutation,
    random_reset_mutation,
    uniform_integer_crossover,
)
from vamos.operators.real import PolynomialMutation, SBXCrossover
from vamos.operators.real import VariationWorkspace


# Operator registries
BINARY_CROSSOVER = {
    "one_point": one_point_crossover,
    "single_point": one_point_crossover,
    "1point": one_point_crossover,
    "two_point": two_point_crossover,
    "2point": two_point_crossover,
    "uniform": uniform_crossover,
}

BINARY_MUTATION = {
    "bitflip": bit_flip_mutation,
    "bit_flip": bit_flip_mutation,
}

INT_CROSSOVER = {
    "uniform": uniform_integer_crossover,
    "blend": arithmetic_integer_crossover,
    "arithmetic": arithmetic_integer_crossover,
}

INT_MUTATION = {
    "reset": random_reset_mutation,
    "random_reset": random_reset_mutation,
    "creep": creep_mutation,
}


def build_variation_operators(
    cfg: dict[str, Any],
    encoding: str,
    n_var: int,
    xl: np.ndarray,
    xu: np.ndarray,
    rng: np.random.Generator,
) -> tuple[Callable, Callable]:
    """Build crossover and mutation operators for the given encoding.

    Parameters
    ----------
    cfg : dict[str, Any]
        Algorithm configuration with 'crossover' and 'mutation' keys.
    encoding : str
        Problem encoding type: "continuous", "real", "binary", or "integer".
    n_var : int
        Number of decision variables.
    xl : np.ndarray
        Lower bounds.
    xu : np.ndarray
        Upper bounds.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    tuple[Callable, Callable]
        (crossover_fn, mutation_fn) functions for generating offspring.

    Raises
    ------
    ValueError
        If the encoding is unsupported or the requested operators are not available.
    """
    cross_method, cross_params = cfg["crossover"]
    cross_params = dict(cross_params)

    mut_method, mut_params = cfg["mutation"]
    mut_params = dict(mut_params)
    if mut_params.get("prob") == "1/n":
        mut_params["prob"] = 1.0 / n_var

    if encoding == "binary":
        return _build_binary_operators(cross_method, cross_params, mut_method, mut_params, n_var, rng)
    elif encoding == "integer":
        return _build_integer_operators(cross_method, cross_params, mut_method, mut_params, n_var, xl, xu, rng)
    elif encoding in {"continuous", "real"}:
        return _build_continuous_operators(cross_params, mut_params, n_var, xl, xu, rng)
    else:
        raise ValueError(f"MOEA/D does not support encoding '{encoding}'.")


def _build_binary_operators(
    cross_method: str,
    cross_params: dict,
    mut_method: str,
    mut_params: dict,
    n_var: int,
    rng: np.random.Generator,
) -> tuple[Callable, Callable]:
    """Build variation operators for binary encoding."""
    if cross_method not in BINARY_CROSSOVER:
        raise ValueError(f"Unsupported MOEA/D crossover '{cross_method}' for binary encoding.")
    if mut_method not in BINARY_MUTATION:
        raise ValueError(f"Unsupported MOEA/D mutation '{mut_method}' for binary encoding.")

    cross_fn = BINARY_CROSSOVER[cross_method]
    cross_prob = float(cross_params.get("prob", 0.9))
    mut_fn = BINARY_MUTATION[mut_method]
    mut_prob = resolve_prob_expression(mut_params.get("prob"), n_var, 1.0 / max(1, n_var))

    def crossover(parents, rng=rng):
        return cross_fn(parents, cross_prob, rng)

    def mutation(X_child, rng=rng):
        return mut_fn(X_child, mut_prob, rng) or X_child

    return crossover, mutation


def _build_integer_operators(
    cross_method: str,
    cross_params: dict,
    mut_method: str,
    mut_params: dict,
    n_var: int,
    xl: np.ndarray,
    xu: np.ndarray,
    rng: np.random.Generator,
) -> tuple[Callable, Callable]:
    """Build variation operators for integer encoding."""
    if cross_method not in INT_CROSSOVER:
        raise ValueError(f"Unsupported MOEA/D crossover '{cross_method}' for integer encoding.")
    if mut_method not in INT_MUTATION:
        raise ValueError(f"Unsupported MOEA/D mutation '{mut_method}' for integer encoding.")

    cross_fn = INT_CROSSOVER[cross_method]
    cross_prob = float(cross_params.get("prob", 0.9))
    mut_fn = INT_MUTATION[mut_method]
    mut_prob = resolve_prob_expression(mut_params.get("prob"), n_var, 1.0 / max(1, n_var))
    step = int(mut_params.get("step", 1))

    def crossover(parents, rng=rng):
        return cross_fn(parents, cross_prob, rng)

    if mut_fn is creep_mutation:

        def mutation(X_child, rng=rng):
            return mut_fn(X_child, mut_prob, step, xl, xu, rng) or X_child
    else:

        def mutation(X_child, rng=rng):
            return mut_fn(X_child, mut_prob, xl, xu, rng) or X_child

    return crossover, mutation


def _build_continuous_operators(
    cross_params: dict,
    mut_params: dict,
    n_var: int,
    xl: np.ndarray,
    xu: np.ndarray,
    rng: np.random.Generator,
) -> tuple[Callable, Callable]:
    """Build variation operators for continuous/real encoding."""
    cross_prob = float(cross_params.get("prob", 0.9))
    cross_eta = float(cross_params.get("eta", 20.0))
    workspace = VariationWorkspace()

    crossover_operator = SBXCrossover(
        prob_crossover=cross_prob,
        eta=cross_eta,
        lower=xl,
        upper=xu,
        workspace=workspace,
        allow_inplace=True,
    )

    mut_prob = resolve_prob_expression(mut_params.get("prob"), n_var, 1.0 / max(1, n_var))
    mut_eta = float(mut_params.get("eta", 20.0))
    mutation_operator = PolynomialMutation(
        prob_mutation=mut_prob,
        eta=mut_eta,
        lower=xl,
        upper=xu,
        workspace=workspace,
    )

    def crossover(parents, rng=rng):
        return crossover_operator(parents, rng)

    def mutation(X_child, rng=rng):
        return mutation_operator(X_child, rng)

    return crossover, mutation


__all__ = [
    "BINARY_CROSSOVER",
    "BINARY_MUTATION",
    "INT_CROSSOVER",
    "INT_MUTATION",
    "build_variation_operators",
]
