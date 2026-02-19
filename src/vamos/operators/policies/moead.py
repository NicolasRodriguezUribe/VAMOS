# operators/policies/moead.py
"""
Operator building for MOEA/D.

This module handles the construction of variation operators (crossover and mutation)
for different encodings (continuous, binary, integer, permutation, mixed).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

import numpy as np

from vamos.engine.algorithm.components.utils import resolve_prob_expression
from vamos.foundation.encoding import EncodingLike, normalize_encoding
from vamos.operators.impl.binary import (
    bit_flip_mutation,
    one_point_crossover,
    two_point_crossover,
    uniform_crossover,
)
from vamos.operators.impl.integer import (
    arithmetic_integer_crossover,
    creep_mutation,
    integer_polynomial_mutation,
    integer_sbx_crossover,
    random_reset_mutation,
    uniform_integer_crossover,
)
from vamos.operators.impl.mixed import mixed_crossover, mixed_mutation
from vamos.operators.impl.permutation import (
    cycle_crossover,
    displacement_mutation,
    edge_recombination_crossover,
    insert_mutation,
    inversion_mutation,
    order_crossover,
    pmx_crossover,
    position_based_crossover,
    scramble_mutation,
    swap_mutation,
)
from vamos.operators.impl.real import PolynomialMutation, SBXCrossover, VariationWorkspace

# Operator registries
BinaryCrossoverOp: TypeAlias = Callable[[np.ndarray, float, np.random.Generator], np.ndarray]
BinaryMutationOp: TypeAlias = Callable[[np.ndarray, float, np.random.Generator], None]
IntCrossoverOp: TypeAlias = Callable[..., np.ndarray]
IntMutationOp: TypeAlias = Callable[..., None]
PermCrossoverOp: TypeAlias = Callable[[np.ndarray, float, np.random.Generator], np.ndarray]
PermMutationOp: TypeAlias = Callable[[np.ndarray, float, np.random.Generator], None]

VariationCrossoverFn: TypeAlias = Callable[[np.ndarray, np.random.Generator], np.ndarray]
VariationMutationFn: TypeAlias = Callable[[np.ndarray, np.random.Generator], np.ndarray]


BINARY_CROSSOVER: dict[str, BinaryCrossoverOp] = {
    "one_point": one_point_crossover,
    "single_point": one_point_crossover,
    "1point": one_point_crossover,
    "spx": one_point_crossover,
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
    "sbx": integer_sbx_crossover,
}

INT_MUTATION: dict[str, IntMutationOp] = {
    "reset": random_reset_mutation,
    "random_reset": random_reset_mutation,
    "creep": creep_mutation,
    "pm": integer_polynomial_mutation,
    "polynomial": integer_polynomial_mutation,
}

PERM_CROSSOVER: dict[str, PermCrossoverOp] = {
    "ox": order_crossover,
    "order": order_crossover,
    "pmx": pmx_crossover,
    "cycle": cycle_crossover,
    "cx": cycle_crossover,
    "position": position_based_crossover,
    "position_based": position_based_crossover,
    "pos": position_based_crossover,
    "edge": edge_recombination_crossover,
    "edge_recombination": edge_recombination_crossover,
    "erx": edge_recombination_crossover,
}

PERM_MUTATION: dict[str, PermMutationOp] = {
    "swap": swap_mutation,
    "insert": insert_mutation,
    "scramble": scramble_mutation,
    "inversion": inversion_mutation,
    "displacement": displacement_mutation,
}


def build_variation_operators(
    cfg: dict[str, Any],
    encoding: EncodingLike,
    n_var: int,
    xl: np.ndarray,
    xu: np.ndarray,
    rng: np.random.Generator,
    mixed_spec: dict[str, np.ndarray] | None = None,
) -> tuple[VariationCrossoverFn, VariationMutationFn]:
    """Build crossover and mutation operators for the given encoding.

    Parameters
    ----------
    cfg : dict[str, Any]
        Algorithm configuration with 'crossover' and 'mutation' keys.
    encoding : str
        Problem encoding type: "real", "binary", "integer", "permutation", or "mixed".
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

    normalized = normalize_encoding(encoding)

    if normalized == "binary":
        return _build_binary_operators(cross_method, cross_params, mut_method, mut_params, n_var, rng)
    elif normalized == "integer":
        return _build_integer_operators(cross_method, cross_params, mut_method, mut_params, n_var, xl, xu, rng)
    elif normalized == "permutation":
        return _build_permutation_operators(cross_method, cross_params, mut_method, mut_params, n_var, rng)
    elif normalized == "mixed":
        if mixed_spec is None:
            raise ValueError("MOEA/D mixed encoding requires problem.mixed_spec.")
        return _build_mixed_operators(
            cross_method,
            cross_params,
            mut_method,
            mut_params,
            n_var,
            mixed_spec,
            rng,
        )
    elif normalized == "real":
        return _build_continuous_operators(cross_method, cross_params, mut_params, n_var, xl, xu, rng)
    else:
        raise ValueError(f"MOEA/D does not support encoding '{normalized}'.")


def _build_binary_operators(
    cross_method: str,
    cross_params: dict[str, Any],
    mut_method: str,
    mut_params: dict[str, Any],
    n_var: int,
    rng: np.random.Generator,
) -> tuple[VariationCrossoverFn, VariationMutationFn]:
    """Build variation operators for binary encoding."""
    if cross_method not in BINARY_CROSSOVER:
        raise ValueError(f"Unsupported MOEA/D crossover '{cross_method}' for binary encoding.")
    if mut_method not in BINARY_MUTATION:
        raise ValueError(f"Unsupported MOEA/D mutation '{mut_method}' for binary encoding.")

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

    if cross_fn is integer_sbx_crossover:
        eta = float(cross_params.get("eta", 20.0))

        def crossover(parents: np.ndarray, _rng: np.random.Generator = rng) -> np.ndarray:
            return integer_sbx_crossover(parents, cross_prob, eta, xl, xu, _rng)

    else:

        def crossover(parents: np.ndarray, _rng: np.random.Generator = rng) -> np.ndarray:
            return cross_fn(parents, cross_prob, _rng)

    if mut_fn is creep_mutation:

        def mutation(X_child: np.ndarray, _rng: np.random.Generator = rng) -> np.ndarray:
            creep_mutation(X_child, mut_prob, step, xl, xu, _rng)
            return X_child

    elif mut_fn is integer_polynomial_mutation:
        eta = float(mut_params.get("eta", 20.0))

        def mutation(X_child: np.ndarray, _rng: np.random.Generator = rng) -> np.ndarray:
            integer_polynomial_mutation(X_child, mut_prob, eta, xl, xu, _rng)
            return X_child

    else:

        def mutation(X_child: np.ndarray, _rng: np.random.Generator = rng) -> np.ndarray:
            mut_fn(X_child, mut_prob, xl, xu, _rng)
            return X_child

    return crossover, mutation


def _build_continuous_operators(
    cross_method: str,
    cross_params: dict[str, Any],
    mut_params: dict[str, Any],
    n_var: int,
    xl: np.ndarray,
    xu: np.ndarray,
    rng: np.random.Generator,
) -> tuple[VariationCrossoverFn, VariationMutationFn]:
    """Build variation operators for continuous/real encoding."""
    method = (cross_method or "sbx").lower()
    workspace = VariationWorkspace()

    if method in {"de", "differential", "differential_evolution"}:
        cr = float(cross_params.get("cr", cross_params.get("CR", 1.0)))
        f = float(cross_params.get("f", cross_params.get("F", 0.5)))

        def crossover(parents: np.ndarray, _rng: np.random.Generator = rng) -> np.ndarray:
            parents_arr = np.asarray(parents)
            if parents_arr.ndim != 3 or parents_arr.shape[1] != 3:
                raise ValueError("DE crossover expects parents shaped (n_pairs, 3, n_var).")
            n_pairs, _, n_vars = parents_arr.shape
            base = parents_arr[:, 2, :]
            p1 = parents_arr[:, 0, :]
            p2 = parents_arr[:, 1, :]
            mutant = base + f * (p1 - p2)
            rand = _rng.random((n_pairs, n_vars))
            mask = rand < cr
            j_rand = _rng.integers(0, n_vars, size=n_pairs)
            mask[np.arange(n_pairs), j_rand] = True
            child = np.where(mask, mutant, base)
            np.clip(child, xl, xu, out=child)
            return child[:, None, :]

    else:
        cross_prob = float(cross_params.get("prob", 0.9))
        cross_eta = float(cross_params.get("eta", 20.0))
        crossover_operator = SBXCrossover(
            prob_crossover=cross_prob,
            eta=cross_eta,
            lower=xl,
            upper=xu,
            workspace=workspace,
            allow_inplace=True,
        )

        def crossover(parents: np.ndarray, _rng: np.random.Generator = rng) -> np.ndarray:
            return crossover_operator(parents, _rng)

    mut_prob = resolve_prob_expression(mut_params.get("prob"), n_var, 1.0 / max(1, n_var))
    mut_eta = float(mut_params.get("eta", 20.0))
    mutation_operator = PolynomialMutation(
        prob_mutation=mut_prob,
        eta=mut_eta,
        lower=xl,
        upper=xu,
        workspace=workspace,
    )

    def mutation(X_child: np.ndarray, _rng: np.random.Generator = rng) -> np.ndarray:
        return mutation_operator(X_child, _rng)

    return crossover, mutation


def _build_permutation_operators(
    cross_method: str,
    cross_params: dict[str, Any],
    mut_method: str,
    mut_params: dict[str, Any],
    n_var: int,
    rng: np.random.Generator,
) -> tuple[VariationCrossoverFn, VariationMutationFn]:
    """Build variation operators for permutation encoding."""
    if cross_method not in PERM_CROSSOVER:
        raise ValueError(f"Unsupported MOEA/D crossover '{cross_method}' for permutation encoding.")
    if mut_method not in PERM_MUTATION:
        raise ValueError(f"Unsupported MOEA/D mutation '{mut_method}' for permutation encoding.")

    cross_fn = PERM_CROSSOVER[cross_method]
    cross_prob = float(cross_params.get("prob", 0.9))
    mut_fn = PERM_MUTATION[mut_method]
    mut_prob = resolve_prob_expression(mut_params.get("prob"), n_var, 1.0 / max(1, n_var))

    def crossover(parents: np.ndarray, _rng: np.random.Generator = rng) -> np.ndarray:
        parents_flat = parents.reshape(-1, parents.shape[-1])
        offspring_flat = cross_fn(parents_flat, cross_prob, _rng)
        return offspring_flat.reshape(parents.shape)

    def mutation(X_child: np.ndarray, _rng: np.random.Generator = rng) -> np.ndarray:
        mut_fn(X_child, mut_prob, _rng)
        return X_child

    return crossover, mutation


def _build_mixed_operators(
    cross_method: str,
    cross_params: dict[str, Any],
    mut_method: str,
    mut_params: dict[str, Any],
    n_var: int,
    mixed_spec: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[VariationCrossoverFn, VariationMutationFn]:
    """Build variation operators for mixed encoding."""
    if str(cross_method).lower() not in {"mixed", "uniform"}:
        raise ValueError(
            f"Unsupported MOEA/D crossover '{cross_method}' for mixed encoding."
        )
    if str(mut_method).lower() not in {"mixed", "gaussian"}:
        raise ValueError(
            f"Unsupported MOEA/D mutation '{mut_method}' for mixed encoding."
        )

    cross_prob = float(cross_params.get("prob", 0.9))
    mut_prob = resolve_prob_expression(
        mut_params.get("prob"),
        n_var,
        1.0 / max(1, n_var),
    )

    def crossover(parents: np.ndarray, _rng: np.random.Generator = rng) -> np.ndarray:
        parents_flat = parents.reshape(-1, parents.shape[-1])
        offspring_flat = mixed_crossover(parents_flat, cross_prob, mixed_spec, _rng)
        return offspring_flat.reshape(parents.shape)

    def mutation(X_child: np.ndarray, _rng: np.random.Generator = rng) -> np.ndarray:
        mixed_mutation(X_child, mut_prob, mixed_spec, _rng)
        return X_child

    return crossover, mutation


__all__ = [
    "BINARY_CROSSOVER",
    "BINARY_MUTATION",
    "INT_CROSSOVER",
    "INT_MUTATION",
    "PERM_CROSSOVER",
    "PERM_MUTATION",
    "build_variation_operators",
]
