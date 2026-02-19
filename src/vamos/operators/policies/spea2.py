# operators/policies/spea2.py
"""
Operator building for SPEA2.

This module handles the construction of variation operators (crossover and mutation)
for different encodings.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

from vamos.foundation.encoding import normalize_encoding
from vamos.operators.impl.mixed import mixed_crossover, mixed_mutation
from vamos.operators.impl.real import PolynomialMutation, SBXCrossover, VariationWorkspace

if TYPE_CHECKING:
    from vamos.foundation.problem.types import ProblemProtocol


VariationFn: TypeAlias = Callable[[np.ndarray, np.random.Generator], np.ndarray]


def build_variation_operators(
    cfg: dict[str, Any],
    encoding: str,
    n_var: int,
    xl: np.ndarray,
    xu: np.ndarray,
    rng: np.random.Generator,
    problem: ProblemProtocol | None = None,
) -> tuple[VariationFn, VariationFn]:
    """Build variation operators for SPEA2.

    Parameters
    ----------
    cfg : dict[str, Any]
        Algorithm configuration with 'crossover' and 'mutation' keys.
    encoding : str
        Problem encoding type.
    n_var : int
        Number of decision variables.
    xl : np.ndarray
        Lower bounds.
    xu : np.ndarray
        Upper bounds.
    rng : np.random.Generator
        Random number generator.
    problem : ProblemProtocol | None
        Problem instance, required for mixed encoding via ``problem.mixed_spec``.

    Returns
    -------
    tuple[Callable, Callable]
        (crossover_fn, mutation_fn) functions for generating offspring.
    """
    # Unpack crossover config (format: ("sbx", {"prob": 0.9, "eta": 20.0}))
    cross_cfg = cfg.get("crossover", ("sbx", {}))
    if isinstance(cross_cfg, tuple):
        cross_method, cross_params = cross_cfg
        cross_params = dict(cross_params) if cross_params else {}
    else:
        cross_params = cross_cfg or {}

    # Unpack mutation config (format: ("pm", {"prob": "1/n", "eta": 20.0}))
    mut_cfg = cfg.get("mutation", ("pm", {}))
    if isinstance(mut_cfg, tuple):
        mut_method, mut_params = mut_cfg
        mut_params = dict(mut_params) if mut_params else {}
    else:
        mut_params = mut_cfg or {}

    # Prepare mutation probability
    mut_prob = mut_params.get("prob", 1.0 / n_var)
    if isinstance(mut_prob, str):
        mut_prob = 1.0 / n_var if "1/n" in mut_prob else float(mut_prob)
    mut_prob = float(mut_prob)

    normalized = normalize_encoding(encoding)
    if normalized == "mixed":
        cross_name = str(cross_method).lower()
        mut_name = str(mut_method).lower()
        if cross_name != "mixed":
            raise ValueError(f"Unsupported SPEA2 crossover '{cross_method}' for mixed encoding. Use 'mixed'.")
        if mut_name != "mixed":
            raise ValueError(f"Unsupported SPEA2 mutation '{mut_method}' for mixed encoding. Use 'mixed'.")
        mixed_spec = getattr(problem, "mixed_spec", None) if problem is not None else None
        if mixed_spec is None:
            raise ValueError("SPEA2 mixed encoding requires problem.mixed_spec.")
        cross_prob = float(cross_params.get("prob", 0.9))

        def crossover_fn(parents: np.ndarray, rng: np.random.Generator = rng) -> np.ndarray:
            parent_shape = parents.shape
            parents_flat = parents.reshape(-1, parent_shape[-1])
            offspring_flat = mixed_crossover(parents_flat, cross_prob, mixed_spec, rng)
            return offspring_flat.reshape(parent_shape)

        def mutation_fn(X_child: np.ndarray, rng: np.random.Generator = rng) -> np.ndarray:
            mixed_mutation(X_child, mut_prob, mixed_spec, rng)
            return X_child

        return crossover_fn, mutation_fn

    workspace = VariationWorkspace()

    # Keep existing SPEA2 behavior for non-mixed encodings.
    crossover_operator = SBXCrossover(
        prob_crossover=cross_params.get("prob", 0.9),
        eta=cross_params.get("eta", 20.0),
        lower=xl,
        upper=xu,
        workspace=workspace,
        allow_inplace=True,
    )
    mutation_operator = PolynomialMutation(
        prob_mutation=float(mut_prob),
        eta=mut_params.get("eta", 20.0),
        lower=xl,
        upper=xu,
        workspace=workspace,
    )

    def crossover_fn(parents: np.ndarray, rng: np.random.Generator = rng) -> np.ndarray:
        return np.asarray(crossover_operator(parents, rng))

    def mutation_fn(X_child: np.ndarray, rng: np.random.Generator = rng) -> np.ndarray:
        return np.asarray(mutation_operator(X_child, rng))

    return crossover_fn, mutation_fn


__all__ = [
    "build_variation_operators",
]
