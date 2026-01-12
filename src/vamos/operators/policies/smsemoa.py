"""SMS-EMOA operator registration and building.

This module provides operator registries and factory functions for SMS-EMOA
supporting continuous, binary, and integer encodings.
"""

from __future__ import annotations

from typing import Any, Callable, TypeAlias

import numpy as np

from vamos.foundation.encoding import EncodingLike, normalize_encoding
from vamos.engine.algorithm.components.utils import resolve_prob_expression
from vamos.operators.impl.binary import (
    bit_flip_mutation,
    one_point_crossover,
    two_point_crossover,
    uniform_crossover,
)
from vamos.operators.impl.integer import (
    arithmetic_integer_crossover,
    creep_mutation,
    random_reset_mutation,
    uniform_integer_crossover,
)
from vamos.operators.impl.real import PolynomialMutation, SBXCrossover
from vamos.operators.impl.real import VariationWorkspace


__all__ = [
    "BINARY_CROSSOVER",
    "BINARY_MUTATION",
    "INT_CROSSOVER",
    "INT_MUTATION",
    "build_variation_operators",
]


# -------------------------------------------------------------------------
# Operator registries
# -------------------------------------------------------------------------

BinaryCrossoverOp: TypeAlias = Callable[[np.ndarray, float, np.random.Generator], np.ndarray]
BinaryMutationOp: TypeAlias = Callable[[np.ndarray, float, np.random.Generator], None]
IntCrossoverOp: TypeAlias = Callable[[np.ndarray, float, np.random.Generator], np.ndarray]
IntMutationOp: TypeAlias = Callable[..., None]

VariationCrossoverFn: TypeAlias = Callable[[np.ndarray], np.ndarray]
VariationMutationFn: TypeAlias = Callable[[np.ndarray], np.ndarray]


BINARY_CROSSOVER: dict[str, BinaryCrossoverOp] = {
    "one_point": one_point_crossover,
    "single_point": one_point_crossover,
    "1point": one_point_crossover,
    "two_point": two_point_crossover,
    "2point": two_point_crossover,
    "uniform": uniform_crossover,
}

BINARY_MUTATION: dict[str, BinaryMutationOp] = {
    "bitflip": bit_flip_mutation,
    "bit_flip": bit_flip_mutation,
}

INT_CROSSOVER: dict[str, IntCrossoverOp] = {
    "uniform": uniform_integer_crossover,
    "blend": arithmetic_integer_crossover,
    "arithmetic": arithmetic_integer_crossover,
}

INT_MUTATION: dict[str, IntMutationOp] = {
    "reset": random_reset_mutation,
    "random_reset": random_reset_mutation,
    "creep": creep_mutation,
}


# -------------------------------------------------------------------------
# Operator builder
# -------------------------------------------------------------------------


def build_variation_operators(
    config: dict[str, Any],
    encoding: EncodingLike,
    n_var: int,
    xl: np.ndarray,
    xu: np.ndarray,
    rng: np.random.Generator,
) -> tuple[VariationCrossoverFn, VariationMutationFn]:
    """Build crossover and mutation operators for the given encoding.

    Parameters
    ----------
    config : dict
        Algorithm configuration containing 'crossover' and 'mutation' keys.
    encoding : str
        Variable encoding: "real", "binary", or "integer".
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
    tuple
        (crossover_fn, mutation_fn) callable operators.

    Raises
    ------
    ValueError
        If encoding or operator is not supported.
    """
    # Unpack crossover config (format: ("sbx", {"prob": 0.9, "eta": 20.0}))
    cross_cfg = config.get("crossover", ("sbx", {}))
    if isinstance(cross_cfg, tuple):
        cross_method, cross_params = cross_cfg
        cross_params = dict(cross_params) if cross_params else {}
    else:
        cross_method = "sbx"
        cross_params = cross_cfg or {}

    # Unpack mutation config (format: ("pm", {"prob": "1/n", "eta": 20.0}))
    mut_cfg = config.get("mutation", ("pm", {}))
    if isinstance(mut_cfg, tuple):
        mut_method, mut_params = mut_cfg
        mut_params = dict(mut_params) if mut_params else {}
    else:
        mut_method = "pm"
        mut_params = mut_cfg or {}

    if mut_params.get("prob") == "1/n":
        mut_params["prob"] = 1.0 / n_var

    normalized = normalize_encoding(encoding)
    if normalized == "binary":
        return _build_binary_operators(cross_method, cross_params, mut_method, mut_params, n_var, rng)
    elif normalized == "integer":
        return _build_integer_operators(cross_method, cross_params, mut_method, mut_params, n_var, xl, xu, rng)
    elif normalized == "real":
        return _build_real_operators(cross_params, mut_params, n_var, xl, xu, rng)
    else:
        raise ValueError(f"SMSEMOA does not support encoding '{normalized}'.")


def _build_binary_operators(
    cross_method: str,
    cross_params: dict[str, Any],
    mut_method: str,
    mut_params: dict[str, Any],
    n_var: int,
    rng: np.random.Generator,
) -> tuple[VariationCrossoverFn, VariationMutationFn]:
    """Build binary encoding operators."""
    if cross_method not in BINARY_CROSSOVER:
        raise ValueError(f"Unsupported SMSEMOA crossover '{cross_method}' for binary encoding.")
    if mut_method not in BINARY_MUTATION:
        raise ValueError(f"Unsupported SMSEMOA mutation '{mut_method}' for binary encoding.")

    cross_fn = BINARY_CROSSOVER[cross_method]
    cross_prob = float(cross_params.get("prob", 0.9))
    mut_fn = BINARY_MUTATION[mut_method]
    mut_prob = resolve_prob_expression(mut_params.get("prob"), n_var, 1.0 / max(1, n_var))

    def crossover(parents: np.ndarray, _rng: np.random.Generator = rng) -> np.ndarray:
        return cross_fn(parents, cross_prob, _rng)

    def mutation(X_child: np.ndarray, _rng: np.random.Generator = rng) -> np.ndarray:
        mut_fn(X_child, mut_prob, _rng)
        return X_child

    return crossover, mutation


def _build_integer_operators(
    cross_method: str,
    cross_params: dict[str, Any],
    mut_method: str,
    mut_params: dict[str, Any],
    n_var: int,
    xl: np.ndarray,
    xu: np.ndarray,
    rng: np.random.Generator,
) -> tuple[VariationCrossoverFn, VariationMutationFn]:
    """Build integer encoding operators."""
    if cross_method not in INT_CROSSOVER:
        raise ValueError(f"Unsupported SMSEMOA crossover '{cross_method}' for integer encoding.")
    if mut_method not in INT_MUTATION:
        raise ValueError(f"Unsupported SMSEMOA mutation '{mut_method}' for integer encoding.")

    cross_fn = INT_CROSSOVER[cross_method]
    cross_prob = float(cross_params.get("prob", 0.9))
    mut_fn = INT_MUTATION[mut_method]
    mut_prob = resolve_prob_expression(mut_params.get("prob"), n_var, 1.0 / max(1, n_var))
    step = int(mut_params.get("step", 1))

    def crossover(parents: np.ndarray, _rng: np.random.Generator = rng) -> np.ndarray:
        return cross_fn(parents, cross_prob, _rng)

    if mut_fn is creep_mutation:

        def mutation(X_child: np.ndarray, _rng: np.random.Generator = rng) -> np.ndarray:
            creep_mutation(X_child, mut_prob, step, xl, xu, _rng)
            return X_child

    else:

        def mutation(X_child: np.ndarray, _rng: np.random.Generator = rng) -> np.ndarray:
            mut_fn(X_child, mut_prob, xl, xu, _rng)
            return X_child

    return crossover, mutation


def _build_real_operators(
    cross_params: dict[str, Any],
    mut_params: dict[str, Any],
    n_var: int,
    xl: np.ndarray,
    xu: np.ndarray,
    rng: np.random.Generator,
) -> tuple[VariationCrossoverFn, VariationMutationFn]:
    """Build continuous (real) encoding operators using SBX and PM."""
    cross_prob = float(cross_params.get("prob", 0.9))
    cross_eta = float(cross_params.get("eta", 20.0))
    workspace = VariationWorkspace()

    sbx = SBXCrossover(
        prob_crossover=cross_prob,
        eta=cross_eta,
        lower=xl,
        upper=xu,
        workspace=workspace,
        allow_inplace=True,
    )

    mut_prob = resolve_prob_expression(mut_params.get("prob"), n_var, 1.0 / max(1, n_var))
    mut_eta = float(mut_params.get("eta", 20.0))
    pm = PolynomialMutation(
        prob_mutation=mut_prob,
        eta=mut_eta,
        lower=xl,
        upper=xu,
        workspace=workspace,
    )

    def crossover(parents: np.ndarray, _rng: np.random.Generator = rng) -> np.ndarray:
        return sbx(parents, _rng)

    def mutation(X_child: np.ndarray, _rng: np.random.Generator = rng) -> np.ndarray:
        return pm(X_child, _rng)

    return crossover, mutation
