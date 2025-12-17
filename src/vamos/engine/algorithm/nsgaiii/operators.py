"""NSGA-III operator registration and building.

This module provides operator registries and factory functions for NSGA-III
supporting continuous, binary, and integer encodings.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from vamos.engine.algorithm.components.utils import resolve_prob_expression
from vamos.engine.operators.binary import (
    bit_flip_mutation,
    one_point_crossover,
    two_point_crossover,
    uniform_crossover,
)
from vamos.engine.operators.integer import (
    arithmetic_integer_crossover,
    creep_mutation,
    random_reset_mutation,
    uniform_integer_crossover,
)
from vamos.engine.operators.real import (
    PolynomialMutation,
    SBXCrossover,
    VariationWorkspace,
)


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

BINARY_CROSSOVER: dict[str, Callable[..., np.ndarray]] = {
    "one_point": one_point_crossover,
    "single_point": one_point_crossover,
    "1point": one_point_crossover,
    "two_point": two_point_crossover,
    "2point": two_point_crossover,
    "uniform": uniform_crossover,
}

BINARY_MUTATION: dict[str, Callable[..., np.ndarray]] = {
    "bitflip": bit_flip_mutation,
    "bit_flip": bit_flip_mutation,
}

INT_CROSSOVER: dict[str, Callable[..., np.ndarray]] = {
    "uniform": uniform_integer_crossover,
    "blend": arithmetic_integer_crossover,
    "arithmetic": arithmetic_integer_crossover,
}

INT_MUTATION: dict[str, Callable[..., np.ndarray]] = {
    "reset": random_reset_mutation,
    "random_reset": random_reset_mutation,
    "creep": creep_mutation,
}


# -------------------------------------------------------------------------
# Operator builder
# -------------------------------------------------------------------------


def build_variation_operators(
    config: dict,
    encoding: str,
    n_var: int,
    xl: np.ndarray,
    xu: np.ndarray,
    rng: np.random.Generator,
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Build crossover and mutation operators for the given encoding.

    Parameters
    ----------
    config : dict
        Algorithm configuration containing 'crossover' and 'mutation' keys.
    encoding : str
        Variable encoding: "continuous", "real", "binary", or "integer".
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
    # Unpack crossover config
    cross_cfg = config.get("crossover", ("sbx", {}))
    if isinstance(cross_cfg, tuple):
        cross_method, cross_params = cross_cfg
        cross_params = dict(cross_params) if cross_params else {}
    else:
        cross_method = "sbx"
        cross_params = cross_cfg or {}

    # Unpack mutation config
    mut_cfg = config.get("mutation", ("pm", {}))
    if isinstance(mut_cfg, tuple):
        mut_method, mut_params = mut_cfg
        mut_params = dict(mut_params) if mut_params else {}
    else:
        mut_method = "pm"
        mut_params = mut_cfg or {}

    if mut_params.get("prob") == "1/n":
        mut_params["prob"] = 1.0 / n_var

    if encoding == "binary":
        return _build_binary_operators(
            cross_method, cross_params, mut_method, mut_params, n_var, rng
        )
    elif encoding == "integer":
        return _build_integer_operators(
            cross_method, cross_params, mut_method, mut_params, n_var, xl, xu, rng
        )
    elif encoding in {"continuous", "real"}:
        return _build_real_operators(cross_params, mut_params, n_var, xl, xu, rng)
    else:
        raise ValueError(f"NSGA-III does not support encoding '{encoding}'.")


def _build_binary_operators(
    cross_method: str,
    cross_params: dict,
    mut_method: str,
    mut_params: dict,
    n_var: int,
    rng: np.random.Generator,
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Build binary encoding operators."""
    if cross_method not in BINARY_CROSSOVER:
        raise ValueError(
            f"Unsupported NSGA-III crossover '{cross_method}' for binary encoding."
        )
    if mut_method not in BINARY_MUTATION:
        raise ValueError(
            f"Unsupported NSGA-III mutation '{mut_method}' for binary encoding."
        )

    cross_fn = BINARY_CROSSOVER[cross_method]
    cross_prob = float(cross_params.get("prob", 0.9))
    mut_fn = BINARY_MUTATION[mut_method]
    mut_prob = resolve_prob_expression(
        mut_params.get("prob"), n_var, 1.0 / max(1, n_var)
    )

    def crossover(parents: np.ndarray, _rng: np.random.Generator = rng) -> np.ndarray:
        return cross_fn(parents, cross_prob, _rng)

    def mutation(X_child: np.ndarray, _rng: np.random.Generator = rng) -> np.ndarray:
        result = mut_fn(X_child, mut_prob, _rng)
        return result if result is not None else X_child

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
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Build integer encoding operators."""
    if cross_method not in INT_CROSSOVER:
        raise ValueError(
            f"Unsupported NSGA-III crossover '{cross_method}' for integer encoding."
        )
    if mut_method not in INT_MUTATION:
        raise ValueError(
            f"Unsupported NSGA-III mutation '{mut_method}' for integer encoding."
        )

    cross_fn = INT_CROSSOVER[cross_method]
    cross_prob = float(cross_params.get("prob", 0.9))
    mut_fn = INT_MUTATION[mut_method]
    mut_prob = resolve_prob_expression(
        mut_params.get("prob"), n_var, 1.0 / max(1, n_var)
    )
    step = int(mut_params.get("step", 1))

    def crossover(parents: np.ndarray, _rng: np.random.Generator = rng) -> np.ndarray:
        return cross_fn(parents, cross_prob, _rng)

    if mut_fn is creep_mutation:

        def mutation(
            X_child: np.ndarray, _rng: np.random.Generator = rng
        ) -> np.ndarray:
            result = mut_fn(X_child, mut_prob, step, xl, xu, _rng)
            return result if result is not None else X_child

    else:

        def mutation(
            X_child: np.ndarray, _rng: np.random.Generator = rng
        ) -> np.ndarray:
            result = mut_fn(X_child, mut_prob, xl, xu, _rng)
            return result if result is not None else X_child

    return crossover, mutation


def _build_real_operators(
    cross_params: dict,
    mut_params: dict,
    n_var: int,
    xl: np.ndarray,
    xu: np.ndarray,
    rng: np.random.Generator,
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
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

    mut_prob = resolve_prob_expression(
        mut_params.get("prob"), n_var, 1.0 / max(1, n_var)
    )
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
