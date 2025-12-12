# algorithm/nsgaii/operators.py
"""
Operator pool building and adaptive operator selection for NSGA-II.

This module handles the construction of variation pipelines and optional
adaptive operator selection mechanisms.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from vamos.algorithm.components.variation import VariationPipeline, prepare_mutation_params
from vamos.hyperheuristics.indicator import IndicatorEvaluator
from vamos.hyperheuristics.operator_selector import make_operator_selector
from vamos.operators.real import VariationWorkspace
from vamos.problem.types import ProblemProtocol


def build_operator_pool(
    cfg: dict[str, Any],
    encoding: str,
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
) -> tuple[list[VariationPipeline], Any | None, IndicatorEvaluator | None]:
    """Build the operator pool and optional adaptive selector.

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

    Returns
    -------
    tuple[list[VariationPipeline], Any | None, IndicatorEvaluator | None]
        (operator_pool, operator_selector, indicator_evaluator)
    """
    operator_pool = _build_variation_pipelines(
        cfg, encoding, cross_method, cross_params, mut_method, mut_params,
        n_var, xl, xu, variation_workspace, problem, mut_factor,
    )
    op_selector, indicator_eval = _setup_adaptive_selection(cfg, len(operator_pool))
    return operator_pool, op_selector, indicator_eval


def _build_variation_pipelines(
    cfg: dict[str, Any],
    encoding: str,
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
    op_configs = cfg.get("adaptive_operators", {}).get("operator_pool")

    if op_configs:
        for entry in op_configs:
            c_method, c_params = entry.get("crossover", (cross_method, cross_params))
            m_method, m_params = entry.get("mutation", (mut_method, mut_params))
            m_params = prepare_mutation_params(m_params, encoding, n_var, prob_factor=mut_factor)
            operator_pool.append(
                _create_variation_pipeline(
                    encoding, c_method, c_params, m_method, m_params,
                    xl, xu, variation_workspace, cfg.get("repair"), problem,
                )
            )

    # Default pipeline if none configured
    if not operator_pool:
        operator_pool.append(
            _create_variation_pipeline(
                encoding, cross_method, cross_params, mut_method, mut_params,
                xl, xu, variation_workspace, cfg.get("repair"), problem,
            )
        )

    return operator_pool


def _create_variation_pipeline(
    encoding: str,
    cross_method: str,
    cross_params: dict[str, Any],
    mut_method: str,
    mut_params: dict[str, Any],
    xl: np.ndarray,
    xu: np.ndarray,
    workspace: VariationWorkspace,
    repair_cfg: Any | None,
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
    repair_cfg : Any | None
        Optional repair configuration.
    problem : ProblemProtocol
        The optimization problem.

    Returns
    -------
    VariationPipeline
        Configured variation pipeline.
    """
    return VariationPipeline(
        encoding=encoding,
        cross_method=cross_method,
        cross_params=cross_params,
        mut_method=mut_method,
        mut_params=mut_params,
        xl=xl,
        xu=xu,
        workspace=workspace,
        repair_cfg=repair_cfg,
        problem=problem,
    )


def _setup_adaptive_selection(
    cfg: dict[str, Any],
    n_operators: int,
) -> tuple[Any | None, IndicatorEvaluator | None]:
    """Setup adaptive operator selection if enabled.

    Parameters
    ----------
    cfg : dict[str, Any]
        Algorithm configuration.
    n_operators : int
        Number of operators in the pool.

    Returns
    -------
    tuple[Any | None, IndicatorEvaluator | None]
        (operator_selector, indicator_evaluator) or (None, None) if disabled.
    """
    selector_cfg = cfg.get("adaptive_operators", {})
    adaptive_enabled = bool(selector_cfg.get("enabled", False)) and n_operators > 1

    if not adaptive_enabled:
        return None, None

    method = selector_cfg.get("method", "epsilon_greedy")
    op_selector = make_operator_selector(
        method,
        n_operators,
        epsilon=selector_cfg.get("epsilon", 0.1),
        c=selector_cfg.get("c", 1.0),
    )
    indicator = selector_cfg.get("indicator", "hv")
    indicator_mode = selector_cfg.get("mode", "maximize")
    indicator_eval = IndicatorEvaluator(indicator, reference_point=None, mode=indicator_mode)

    return op_selector, indicator_eval
