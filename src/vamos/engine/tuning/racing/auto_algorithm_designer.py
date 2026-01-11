"""
Generic Auto-Solver: Adaptive Algorithm Design via Meta-Learning.

Demonstrates VAMOS as a meta-learning framework that automatically designs
algorithm configurations based on problem encoding (real, binary, permutation).

Usage:
    python examples/auto_design/generic_auto_solver.py zdt1
    python examples/auto_design/generic_auto_solver.py bin_knapsack
    python examples/auto_design/generic_auto_solver.py tsp6
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Callable, cast

import numpy as np

from vamos.api import OptimizeConfig, optimize
from vamos.engine.api import NSGAIIConfig, NSGAIIConfigData
from vamos.engine.tuning.api import (
    Categorical,
    Condition,
    EvalContext,
    Instance,
    Int,
    ParamSpace,
    RacingTuner,
    Real,
    Scenario,
    TuningTask,
)
from vamos.engine.config.variation import (
    get_crossover_names,
    get_mutation_names,
    get_operators_for_encoding,
)
from vamos.foundation.problem.registry import get_problem_specs
from vamos.foundation.problem.types import ProblemProtocol


# =============================================================================
# Conditional Hyperparameters Registry
# =============================================================================


def _conditional_params() -> dict[str, dict[str, tuple[str, Real]]]:
    return {
        "crossover": {
            "sbx": ("sbx_eta", Real("sbx_eta", 5.0, 30.0)),
            "blx": ("blx_alpha", Real("blx_alpha", 0.1, 0.5)),
            "de": ("de_F", Real("de_F", 0.3, 0.9)),
        },
        "mutation": {
            "pm": ("pm_eta", Real("pm_eta", 5.0, 30.0)),
            "gaussian": ("gauss_sigma", Real("gauss_sigma", 0.01, 0.3)),
        },
    }


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def _add_conditional_params(
    params: dict[str, Any],
    conditions: list[Condition],
    operator_type: str,
    available_ops: dict[str, Any],
) -> None:
    """Add conditional hyperparameters for operators that require them."""
    conditional_params = _conditional_params()
    for op_name, (param_name, param_def) in conditional_params.get(operator_type, {}).items():
        if op_name in available_ops:
            params[param_name] = param_def
            conditions.append(Condition(param_name, f"cfg['{operator_type}'] == '{op_name}'"))


# =============================================================================
# Dynamic Search Space Construction
# =============================================================================


def build_param_space_for_encoding(encoding: str) -> ParamSpace:
    """
    Dynamically build a hyperparameter space based on problem encoding.

    Operators are auto-detected from the OPERATORS_BY_ENCODING registry.
    Conditional hyperparameters are added for operators that require them.
    """
    # Normalize encoding
    if encoding == "continuous":
        encoding = "real"

    # Get operators from registry
    crossover_names = get_crossover_names(encoding)
    mutation_names = get_mutation_names(encoding)
    operators_info = get_operators_for_encoding(encoding)

    params: dict[str, Any] = {
        # Common parameters across all encodings
        "pop_size": Int("pop_size", 50, 150, log=True),
        "crossover_prob": Real("crossover_prob", 0.6, 0.95),
        "mutation_prob": Real("mutation_prob", 0.01, 0.3),
        # Auto-detected operators
        "crossover": Categorical("crossover", crossover_names),
        "mutation": Categorical("mutation", mutation_names),
        # Result source: population vs external archive
        "result_mode": Categorical("result_mode", ["population", "archive"]),
    }
    conditions: list[Condition] = []

    # Archive configuration (only active when result_mode == "archive")
    params["archive_type"] = Categorical("archive_type", ["size_cap", "epsilon_grid", "hvc_prune", "hybrid"])
    conditions.append(Condition("archive_type", "cfg['result_mode'] == 'archive'"))

    # Archive size cap
    params["archive_size"] = Int("archive_size", 50, 300)
    conditions.append(Condition("archive_size", "cfg['result_mode'] == 'archive'"))

    # Epsilon for grid-based archives (will be ignored for non-epsilon types)
    params["archive_epsilon"] = Real("archive_epsilon", 0.001, 0.1, log=True)
    conditions.append(Condition("archive_epsilon", "cfg['result_mode'] == 'archive'"))

    # Prune policy for bounded archives
    params["prune_policy"] = Categorical("prune_policy", ["crowding", "hv_contrib", "random"])
    conditions.append(Condition("prune_policy", "cfg['result_mode'] == 'archive'"))

    # Add conditional hyperparameters from registry
    crossover_ops = {op[0]: op[1] for op in operators_info.get("crossover", [])}
    mutation_ops = {op[0]: op[1] for op in operators_info.get("mutation", [])}

    _add_conditional_params(params, conditions, "crossover", crossover_ops)
    _add_conditional_params(params, conditions, "mutation", mutation_ops)

    return ParamSpace(params=params, conditions=conditions)


# =============================================================================
# Algorithm Configuration Translation
# =============================================================================


def make_algo_config(assignment: dict[str, Any], encoding: str) -> NSGAIIConfigData:
    """Translate hyperparameters into an NSGA-II config based on encoding."""
    config = NSGAIIConfig().pop_size(int(assignment["pop_size"])).offspring_size(int(assignment["pop_size"]))

    # Crossover configuration
    crossover_type = assignment.get("crossover", "sbx")
    crossover_prob = float(assignment["crossover_prob"])

    if crossover_type == "sbx":
        eta = float(assignment.get("sbx_eta", 20.0))
        config = config.crossover("sbx", prob=crossover_prob, eta=eta)
    elif crossover_type == "uniform":
        config = config.crossover("uniform", prob=crossover_prob)
    elif crossover_type == "de":
        F = float(assignment.get("de_F", 0.5))
        config = config.crossover("de", prob=crossover_prob, F=F)
    elif crossover_type == "blx":
        alpha = float(assignment.get("blx_alpha", 0.5))
        config = config.crossover("blx", prob=crossover_prob, alpha=alpha)
    elif crossover_type == "hux":
        config = config.crossover("hux", prob=crossover_prob)
    elif crossover_type in ("pmx", "ox", "edge", "cycle", "position"):
        config = config.crossover(crossover_type, prob=crossover_prob)
    else:
        config = config.crossover(crossover_type, prob=crossover_prob)

    # Mutation configuration
    mutation_type = assignment.get("mutation", "pm")
    mutation_prob = float(assignment["mutation_prob"])

    if mutation_type == "pm":
        eta = float(assignment.get("pm_eta", 20.0))
        config = config.mutation("pm", prob=mutation_prob, eta=eta)
    elif mutation_type == "gaussian":
        sigma = float(assignment.get("gauss_sigma", 0.1))
        config = config.mutation("gaussian", prob=mutation_prob, sigma=sigma)
    elif mutation_type == "bitflip":
        config = config.mutation("bitflip", prob=mutation_prob)
    elif mutation_type in ("swap", "inversion", "scramble", "insert", "displacement"):
        config = config.mutation(mutation_type, prob=mutation_prob)
    else:
        config = config.mutation(mutation_type, prob=mutation_prob)

    # Result mode: population vs external archive
    result_mode = assignment.get("result_mode", "population")
    if result_mode == "archive":
        archive_type = assignment.get("archive_type", "size_cap")
        archive_size = int(assignment.get("archive_size", 200))
        prune_policy = assignment.get("prune_policy", "crowding")

        archive_kwargs = {
            "archive_type": archive_type,
            "prune_policy": prune_policy,
        }

        # Add epsilon for grid-based archives
        if archive_type in ("epsilon_grid", "hybrid"):
            archive_kwargs["epsilon"] = float(assignment.get("archive_epsilon", 0.01))

        config = config.result_mode("external_archive").archive(archive_size, **archive_kwargs)
    else:
        config = config.result_mode("population")

    return config.selection("tournament", pressure=2).engine("numpy").fixed()


# =============================================================================
# Evaluation Function
# =============================================================================


def make_evaluator(problem_name: str, encoding: str) -> Callable[[dict[str, Any], EvalContext], float]:
    """Create an evaluation function for the tuner."""
    specs = get_problem_specs()

    def evaluate_config(config: dict[str, Any], ctx: EvalContext) -> float:
        spec = specs[problem_name]
        problem = cast(ProblemProtocol, spec.factory(spec.default_n_var, spec.default_n_obj))

        algo_cfg = make_algo_config(config, encoding)

        try:
            result = optimize(
                OptimizeConfig(
                    problem=problem,
                    algorithm="nsgaii",
                    algorithm_config=algo_cfg,
                    termination=("n_eval", ctx.budget),
                    seed=ctx.seed,
                    engine="numpy",
                )
            )
            F = result.F
            if F is None or len(F) == 0:
                return float("inf")
            # Minimize sum of objectives (simple aggregation)
            return float(np.mean(np.sum(F, axis=1)))
        except Exception as e:
            _logger().warning("Evaluation failed: %s", e)
            return float("inf")

    return evaluate_config


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generic Auto-Solver: Adaptive Algorithm Design via Meta-Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generic_auto_solver.py zdt1          # Real encoding (continuous)
    python generic_auto_solver.py bin_knapsack  # Binary encoding
    python generic_auto_solver.py tsp6          # Permutation encoding
        """,
    )
    parser.add_argument("problem", help="Problem name from VAMOS registry")
    parser.add_argument("--budget", type=int, default=1000, help="Evaluations per tuning run")
    parser.add_argument("--max-configs", type=int, default=8, help="Max configurations to evaluate")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    logger = _logger()

    # Load problem and detect encoding
    specs = get_problem_specs()
    if args.problem not in specs:
        logger.error("Problem '%s' not found in registry.", args.problem)
        logger.info("Available: %s...", ", ".join(sorted(specs.keys())[:20]))
        sys.exit(1)

    spec = specs[args.problem]
    problem = spec.factory(spec.default_n_var, spec.default_n_obj)
    encoding = getattr(problem, "encoding", spec.encoding if hasattr(spec, "encoding") else "real")

    logger.info("%s", "=" * 60)
    logger.info(" VAMOS Generic Auto-Solver: Meta-Learning Framework")
    logger.info("%s", "=" * 60)
    logger.info("  Problema: %s", args.problem)
    logger.info("  Encoding detectado: %s", encoding)
    logger.info("  Variables: %s, Objetivos: %s", spec.default_n_var, spec.default_n_obj)
    logger.info("%s", "=" * 60)

    # Build dynamic parameter space
    param_space = build_param_space_for_encoding(encoding)
    logger.info("[*] Espacio de búsqueda dinámico (%s parámetros):", len(param_space.params))
    for name, p in param_space.params.items():
        if isinstance(p, Categorical):
            logger.info("    - %s: %s", name, p.choices)
        elif isinstance(p, (Real, Int)):
            logger.info("    - %s: [%s, %s]", name, p.low, p.high)

    # Setup tuning task
    task = TuningTask(
        name=f"auto_{args.problem}",
        param_space=param_space,
        instances=[Instance(name=args.problem, n_var=spec.default_n_var)],
        seeds=[0, 1],
        budget_per_run=args.budget,
        maximize=False,
    )

    scenario = Scenario(
        max_experiments=args.max_configs * 2,
        min_survivors=2,
        elimination_fraction=0.5,
        start_instances=1,
        verbose=args.verbose,
    )

    logger.info("[*] Iniciando Racing Tuner (máx %s configs)...", args.max_configs)

    tuner = RacingTuner(task=task, scenario=scenario, seed=42, max_initial_configs=args.max_configs)
    evaluator = make_evaluator(args.problem, encoding)
    best_config, history = tuner.run(evaluator, verbose=args.verbose)

    # Format output
    logger.info("%s", "=" * 60)
    logger.info(" Problema detectado: %s", encoding.upper())
    logger.info("%s", "=" * 60)
    logger.info(" Mejor Arquitectura encontrada:")
    logger.info("%s", "-" * 60)
    for k, v in sorted(best_config.items()):
        if isinstance(v, float):
            logger.info("    %s: %.4f", k, v)
        else:
            logger.info("    %s: %s", k, v)

    best_score = min(trial.score for trial in history if trial.score < float("inf"))
    logger.info("%s", "-" * 60)
    logger.info(" Score: %.6f (menor es mejor)", best_score)
    logger.info("%s", "=" * 60)

    # Summary message
    logger.info(
        ">>> Problema detectado: %s -> Mejor Arquitectura encontrada: %s",
        encoding.upper(),
        best_config,
    )


if __name__ == "__main__":
    main()
