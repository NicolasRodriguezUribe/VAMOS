# operators/policies/nsgaii.py
"""
Operator pool building and adaptive operator selection for NSGA-II.

This module handles the construction of variation pipelines and optional
adaptive operator selection mechanisms.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vamos.engine.algorithm.components.variation import VariationPipeline, prepare_mutation_params
from vamos.engine.hyperheuristics.indicator import IndicatorEvaluator
from vamos.engine.hyperheuristics.operator_selector import make_operator_selector
from vamos.operators.impl.real import VariationWorkspace
from vamos.foundation.problem.types import ProblemProtocol
from vamos.adaptation.aos.config import AdaptiveOperatorSelectionConfig
from vamos.adaptation.aos.controller import AOSController
from vamos.adaptation.aos.policies import (
    EpsGreedyPolicy,
    EXP3Policy,
    SlidingWindowUCBPolicy,
    ThompsonSamplingPolicy,
    UCBPolicy,
)
from vamos.adaptation.aos.portfolio import OperatorPortfolio


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
) -> tuple[list[VariationPipeline], Any | None, IndicatorEvaluator | None, AOSController | None]:
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
    tuple[list[VariationPipeline], Any | None, IndicatorEvaluator | None, AOSController | None]
        (operator_pool, operator_selector, indicator_evaluator, aos_controller)
    """
    operator_pool = _build_variation_pipelines(
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
    op_selector, indicator_eval = _setup_adaptive_selection(cfg, len(operator_pool))
    aos_controller = _setup_aos_controller(cfg, operator_pool)
    return operator_pool, op_selector, indicator_eval, aos_controller


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
    aos_cfg = cfg.get("adaptive_operator_selection") or {}
    op_configs = aos_cfg.get("operator_pool") or cfg.get("adaptive_operators", {}).get("operator_pool")

    if op_configs:
        for entry in op_configs:
            c_method, c_params = entry.get("crossover", (cross_method, cross_params))
            m_method, m_params = entry.get("mutation", (mut_method, mut_params))
            m_params = prepare_mutation_params(m_params, encoding, n_var, prob_factor=mut_factor)
            operator_pool.append(
                _create_variation_pipeline(
                    encoding,
                    c_method,
                    c_params,
                    m_method,
                    m_params,
                    xl,
                    xu,
                    variation_workspace,
                    cfg.get("repair"),
                    problem,
                )
            )

    # Default pipeline if none configured
    if not operator_pool:
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
                cfg.get("repair"),
                problem,
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


def _setup_aos_controller(
    cfg: dict[str, Any],
    operator_pool: list[VariationPipeline],
) -> AOSController | None:
    selector_cfg = cfg.get("adaptive_operator_selection")
    aos_config = AdaptiveOperatorSelectionConfig.from_dict(selector_cfg)
    if not aos_config.enabled or len(operator_pool) <= 1:
        return None

    policy_name = aos_config.method
    reward_scope = aos_config.reward_scope
    valid_scopes = {
        "survival",
        "survival_rate",
        "nd",
        "nd_insertion",
        "nd_insertions",
        "hv",
        "hv_delta",
        "hypervolume",
        "combined",
    }
    if reward_scope not in valid_scopes:
        raise ValueError(f"Unsupported reward_scope '{reward_scope}'.")

    if policy_name == "epsilon_greedy":
        policy = EpsGreedyPolicy(
            len(operator_pool),
            epsilon=aos_config.epsilon,
            rng_seed=aos_config.rng_seed,
            min_usage=aos_config.min_usage,
        )
    elif policy_name == "ucb":
        policy = UCBPolicy(
            len(operator_pool),
            c=aos_config.c,
            min_usage=aos_config.min_usage,
        )
    elif policy_name == "exp3":
        policy = EXP3Policy(
            len(operator_pool),
            gamma=aos_config.gamma,
            rng_seed=aos_config.rng_seed,
        )
    elif policy_name == "thompson_sampling":
        policy = ThompsonSamplingPolicy(
            len(operator_pool),
            rng_seed=aos_config.rng_seed,
            min_usage=aos_config.min_usage,
            window_size=aos_config.window_size,
        )
    elif policy_name == "sliding_ucb":
        if aos_config.window_size <= 0:
            raise ValueError("sliding_ucb requires window_size > 0.")
        policy = SlidingWindowUCBPolicy(
            len(operator_pool),
            c=aos_config.c,
            min_usage=aos_config.min_usage,
            window_size=aos_config.window_size,
        )
    else:
        raise ValueError(f"Unsupported AOS method '{policy_name}'.")

    pairs = []
    for idx, pipeline in enumerate(operator_pool):
        op_name = f"{pipeline.cross_method}+{pipeline.mut_method}"
        pairs.append((str(idx), op_name))
    portfolio = OperatorPortfolio.from_pairs(pairs)

    return AOSController(config=aos_config, portfolio=portfolio, policy=policy)
