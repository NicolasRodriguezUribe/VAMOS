# operators/policies/nsgaii.py
"""
Operator pool building for NSGA-II.

This module handles the construction of variation pipelines.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from vamos.engine.operators.impl.real import VariationWorkspace
from vamos.engine.variation import VariationPipeline, prepare_mutation_params
from vamos.engine.variation.protocol import CrossoverName, MutationName, RepairConfigValue
from vamos.foundation.encoding import EncodingLike
from vamos.foundation.problem.types import ProblemProtocol


def build_operator_pool(
    cfg: dict[str, Any],
    encoding: EncodingLike,
    cross_method: str,
    cross_params: dict[str, Any],
    mut_method: str,
    mut_params: dict[str, Any],
    n_var: int,
    xl: np.ndarray,
    xu: np.ndarray,
    variation_workspace: VariationWorkspace,
    problem: ProblemProtocol,
    mut_factor: float | None,
) -> list[VariationPipeline]:
    """Build the operator pool.

    Parameters
    ----------
    cfg : dict[str, Any]
        Algorithm configuration dict.
    encoding : str
        Problem encoding type.
    cross_method : str
        Default crossover method.
    cross_params : dict[str, Any]
        Default crossover parameters.
    mut_method : str
        Default mutation method.
    mut_params : dict[str, Any]
        Default mutation parameters.
    n_var : int
        Number of decision variables.
    xl : np.ndarray
        Lower bounds.
    xu : np.ndarray
        Upper bounds.
    variation_workspace : VariationWorkspace
        Shared workspace for variation operators.
    problem : ProblemProtocol
        The optimization problem.
    mut_factor : float | None
        Optional mutation probability factor.

    Returns:
        operator_pool
    """
    return _build_variation_pipelines(
        cfg,
        encoding,
        cross_method,
        cross_params,
        mut_method,
        mut_params,
        n_var,
        xl,
        xu,
        variation_workspace,
        problem,
        mut_factor,
    )


def _build_variation_pipelines(
    cfg: dict[str, Any],
    encoding: EncodingLike,
    cross_method: str,
    cross_params: dict[str, Any],
    mut_method: str,
    mut_params: dict[str, Any],
    n_var: int,
    xl: np.ndarray,
    xu: np.ndarray,
    variation_workspace: VariationWorkspace,
    problem: ProblemProtocol,
    mut_factor: float | None,
) -> list[VariationPipeline]:
    """Build list of variation pipelines from config.

    Parameters
    ----------
    cfg : dict[str, Any]
        Algorithm configuration.
    encoding : str
        Problem encoding type.
    cross_method : str
        Default crossover method.
    cross_params : dict[str, Any]
        Default crossover parameters.
    mut_method : str
        Default mutation method.
    mut_params : dict[str, Any]
        Default mutation parameters.
    n_var : int
        Number of decision variables.
    xl : np.ndarray
        Lower bounds.
    xu : np.ndarray
        Upper bounds.
    variation_workspace : VariationWorkspace
        Shared workspace.
    problem : ProblemProtocol
        The optimization problem.
    mut_factor : float | None
        Optional mutation probability factor.

    Returns
    -------
    list[VariationPipeline]
        List of configured variation pipelines.
    """
    operator_pool: list[VariationPipeline] = []

    # Default pipeline
    operator_pool.append(
        _create_variation_pipeline(
            encoding,
            cross_method,
            cross_params,
            mut_method,
            mut_params,
            xl,
            xu,
            variation_workspace,
            cfg.get("repair", "auto"),
            problem,
        )
    )

    return operator_pool


def _create_variation_pipeline(
    encoding: EncodingLike,
    cross_method: str,
    cross_params: dict[str, Any],
    mut_method: str,
    mut_params: dict[str, Any],
    xl: np.ndarray,
    xu: np.ndarray,
    workspace: VariationWorkspace,
    repair_cfg: Any,
    problem: ProblemProtocol,
) -> VariationPipeline:
    """Create a single variation pipeline.

    Parameters
    ----------
    encoding : str
        Problem encoding type.
    cross_method : str
        Crossover method name.
    cross_params : dict[str, Any]
        Crossover parameters.
    mut_method : str
        Mutation method name.
    mut_params : dict[str, Any]
        Mutation parameters.
    xl : np.ndarray
        Lower bounds.
    xu : np.ndarray
        Upper bounds.
    workspace : VariationWorkspace
        Shared workspace.
    repair_cfg : Any
        Repair configuration or "auto".
    problem : ProblemProtocol
        The optimization problem.

    Returns
    -------
    VariationPipeline
        Configured variation pipeline.
    """
    return VariationPipeline(
        encoding=encoding,
        cross_method=cast(CrossoverName, str(cross_method).lower()),
        cross_params=cross_params,
        mut_method=cast(MutationName, str(mut_method).lower()),
        mut_params=mut_params,
        xl=xl,
        xu=xu,
        workspace=workspace,
        repair_cfg=cast(RepairConfigValue, repair_cfg),
        problem=problem,
    )
