# algorithm/ibea/operators.py
"""
Operator building for IBEA.

This module handles the construction of variation pipelines.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.engine.algorithm.components.variation import VariationPipeline, prepare_mutation_params
from vamos.engine.operators.real import VariationWorkspace

if TYPE_CHECKING:
    from vamos.foundation.problem.types import ProblemProtocol


def build_variation_pipeline(
    cfg: dict[str, Any],
    encoding: str,
    n_var: int,
    xl: np.ndarray,
    xu: np.ndarray,
    problem: "ProblemProtocol",
) -> VariationPipeline:
    """Build variation pipeline for IBEA.

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
    problem : ProblemProtocol
        The optimization problem.

    Returns
    -------
    VariationPipeline
        Configured variation pipeline.
    """
    cross_method, cross_params = cfg["crossover"]
    cross_method = cross_method.lower()
    cross_params = dict(cross_params)

    mut_method, mut_params = cfg["mutation"]
    mut_method = mut_method.lower()
    mut_params = prepare_mutation_params(
        mut_params, encoding, n_var, prob_factor=cfg.get("mutation_prob_factor")
    )

    variation_workspace = VariationWorkspace()
    variation = VariationPipeline(
        encoding=encoding,
        cross_method=cross_method,
        cross_params=cross_params,
        mut_method=mut_method,
        mut_params=mut_params,
        xl=xl,
        xu=xu,
        workspace=variation_workspace,
        repair_cfg=cfg.get("repair"),
        problem=problem,
    )

    return variation


__all__ = [
    "build_variation_pipeline",
]
