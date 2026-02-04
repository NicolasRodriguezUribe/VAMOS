"""
Generic racing tuner for multi-objective algorithms.

This example selects the algorithm at runtime, builds the appropriate
configuration space, and runs the racing tuner with optional multi-fidelity
and warm-start support.

Usage:
    python examples/tuning/racing_tuner_generic.py --algorithm nsgaii
    python examples/tuning/racing_tuner_generic.py --algorithm moead --multi-fidelity
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Mapping

import numpy as np

from vamos import optimize
from vamos.engine.tuning.api import (
    AlgorithmConfigSpace,
    Instance,
    ParamSpace,
    RacingTuner,
    Scenario,
    TuningTask,
    build_ibea_config_space,
    build_ibea_binary_config_space,
    build_ibea_integer_config_space,
    build_moead_config_space,
    build_moead_binary_config_space,
    build_moead_integer_config_space,
    build_moead_permutation_config_space,
    build_nsgaii_config_space,
    build_nsgaii_binary_config_space,
    build_nsgaii_integer_config_space,
    build_nsgaii_mixed_config_space,
    build_nsgaii_permutation_config_space,
    build_nsgaiii_config_space,
    build_nsgaiii_binary_config_space,
    build_nsgaiii_integer_config_space,
    build_agemoea_config_space,
    build_rvea_config_space,
    build_smsemoa_config_space,
    build_smsemoa_binary_config_space,
    build_smsemoa_integer_config_space,
    build_smpso_config_space,
    build_spea2_config_space,
    config_from_assignment,
)
from vamos.engine.tuning.racing import EvalContext, WarmStartEvaluator
from vamos.foundation.problem.registry import make_problem_selection
from vamos.foundation.metrics.hypervolume import compute_hypervolume


BUILDERS: dict[str, Callable[[], AlgorithmConfigSpace | ParamSpace]] = {
    "nsgaii": build_nsgaii_config_space,
    "nsgaii_permutation": build_nsgaii_permutation_config_space,
    "nsgaii_mixed": build_nsgaii_mixed_config_space,
    "nsgaii_binary": build_nsgaii_binary_config_space,
    "nsgaii_integer": build_nsgaii_integer_config_space,
    "moead": build_moead_config_space,
    "moead_permutation": build_moead_permutation_config_space,
    "moead_binary": build_moead_binary_config_space,
    "moead_integer": build_moead_integer_config_space,
    "nsgaiii": build_nsgaiii_config_space,
    "nsgaiii_binary": build_nsgaiii_binary_config_space,
    "nsgaiii_integer": build_nsgaiii_integer_config_space,
    "spea2": build_spea2_config_space,
    "ibea": build_ibea_config_space,
    "ibea_binary": build_ibea_binary_config_space,
    "ibea_integer": build_ibea_integer_config_space,
    "smpso": build_smpso_config_space,
    "smsemoa": build_smsemoa_config_space,
    "smsemoa_binary": build_smsemoa_binary_config_space,
    "smsemoa_integer": build_smsemoa_integer_config_space,
    "agemoea": build_agemoea_config_space,
    "rvea": build_rvea_config_space,
}


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def _configure_cli_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
    root.setLevel(level)


def _canonical_algorithm_name(name: str) -> str:
    if name in {"nsgaii_permutation", "nsgaii_mixed", "nsgaii_binary", "nsgaii_integer"}:
        return "nsgaii"
    if name in {"moead_permutation", "moead_binary", "moead_integer"}:
        return "moead"
    if name in {"nsgaiii_binary", "nsgaiii_integer"}:
        return "nsgaiii"
    if name in {"smsemoa_binary", "smsemoa_integer"}:
        return "smsemoa"
    if name in {"ibea_binary", "ibea_integer"}:
        return "ibea"
    return name


def _supports_warm_start(name: str) -> bool:
    return _canonical_algorithm_name(name) in {"nsgaii", "moead"}


def _parse_fidelity_levels(raw: str | None, parser: argparse.ArgumentParser) -> tuple[int, ...] | None:
    if raw is None:
        return None
    parts = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    if len(parts) < 2:
        parser.error("--fidelity-levels must provide at least two comma-separated integers.")
    try:
        levels = tuple(int(part) for part in parts)
    except ValueError:
        parser.error("--fidelity-levels must be a comma-separated list of integers (e.g., 500,1000,1500).")
    if any(level <= 0 for level in levels):
        parser.error("--fidelity-levels must be positive integers.")
    return levels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generic VAMOS racing tuner")
    parser.add_argument("--problem", type=str, default="zdt1", help="Problem ID (e.g., zdt1)")
    parser.add_argument("--algorithm", type=str, default="nsgaii", choices=list(BUILDERS.keys()), help="Algorithm to tune")
    parser.add_argument("--n-var", type=int, default=30, help="Number of variables")
    parser.add_argument("--n-obj", type=int, default=2, help="Number of objectives")
    parser.add_argument("--budget", type=int, default=1500, help="Max evaluations per run")
    parser.add_argument(
        "--tune-budget",
        type=int,
        default=20,
        help="Max number of configuration evaluations (config x instance x seed)",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--n-seeds", type=int, default=3, help="Number of seeds to evaluate per config")
    parser.add_argument("--pop-size", type=int, default=100, help="Fixed population size (if not tuning it)")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs (-1 for all cores)")
    parser.add_argument("--ref-point", type=str, default=None, help="Reference point for HV (comma-separated, e.g. '1.1,1.1')")
    parser.add_argument("--multi-fidelity", action="store_true", help="Enable multi-fidelity successive halving")
    parser.add_argument(
        "--fidelity-levels",
        type=str,
        default=None,
        help="Comma-separated budgets for multi-fidelity levels (e.g., 500,1000,1500)",
    )
    parser.add_argument(
        "--fidelity-promotion-ratio",
        type=float,
        default=None,
        help="Fraction of configs promoted to the next fidelity level",
    )
    parser.add_argument(
        "--fidelity-min-configs",
        type=int,
        default=None,
        help="Minimum number of configs to keep at each fidelity level",
    )
    parser.add_argument(
        "--no-warm-start",
        dest="warm_start",
        action="store_false",
        help="Disable warm-start even if the algorithm supports it",
    )
    parser.set_defaults(warm_start=True)
    args = parser.parse_args()
    args.fidelity_levels = _parse_fidelity_levels(args.fidelity_levels, parser)
    return args


def make_evaluator(
    problem_key: str,
    n_var: int,
    n_obj: int,
    algorithm_name: str,
    fixed_pop_size: int,
    ref_point_str: str | None,
    warm_start: bool,
) -> Callable[[Mapping[str, object], EvalContext], float]:
    # Parse reference point once
    ref_point: list[float]
    if ref_point_str:
        try:
            ref_point = [float(x.strip()) for x in ref_point_str.split(",")]
            if len(ref_point) != n_obj:
                _logger().warning(
                    "Reference point length (%s) does not match n_obj (%s).",
                    len(ref_point),
                    n_obj,
                )
        except ValueError:
            _logger().warning("Error parsing --ref-point. Using default.")
            ref_point = [1.1] * n_obj
    else:
        ref_point = [1.1] * n_obj

    def _score(result, ctx: EvalContext) -> float:
        F = getattr(result, "F", None)
        if F is None or len(F) == 0:
            return 0.0
        hv = compute_hypervolume(F, ref_point)
        return float(hv)

    def _run_algorithm(config_dict: Mapping[str, object], ctx: EvalContext, checkpoint: object | None):
        try:
            start_config: dict[str, object] = dict(config_dict)
            if algorithm_name == "rvea":
                start_config["n_obj"] = n_obj
            elif "pop_size" not in start_config:
                start_config["pop_size"] = fixed_pop_size

            cfg = config_from_assignment(algorithm_name, start_config)
            algo_name = _canonical_algorithm_name(algorithm_name)

            selection = make_problem_selection(problem_key, n_var=n_var, n_obj=n_obj)
            result = optimize(
                selection.instantiate(),
                algorithm=algo_name,
                algorithm_config=cfg,
                termination=("n_eval", ctx.budget),
                seed=ctx.seed,
                engine="numpy",
                checkpoint=checkpoint,
            )
            return result, result.data.get("checkpoint")
        except Exception as exc:
            _logger().warning("Eval failed for %s: %s", algorithm_name, exc)

            class _EmptyResult:
                F = None

            return _EmptyResult(), None

    if warm_start:
        return WarmStartEvaluator(run_fn=_run_algorithm, score_fn=_score)

    def eval_fn(config_dict: Mapping[str, object], ctx: EvalContext) -> float:
        result, _ = _run_algorithm(config_dict, ctx, None)
        return _score(result, ctx)

    return eval_fn


def main() -> None:
    _configure_cli_logging()
    args = parse_args()

    builder = BUILDERS[args.algorithm]
    algo_space = builder()
    param_space = algo_space.to_param_space() if isinstance(algo_space, AlgorithmConfigSpace) else algo_space

    fidelity_levels = args.fidelity_levels
    budget_per_run = args.budget
    if args.multi_fidelity and fidelity_levels is not None:
        budget_per_run = max(fidelity_levels)

    warm_start = bool(args.warm_start)
    if warm_start and not _supports_warm_start(args.algorithm):
        _logger().warning("Warm-start is not supported for %s; disabling warm-start.", args.algorithm)
        warm_start = False

    scenario_kwargs = {
        "max_experiments": args.tune_budget,
        "initial_budget_per_run": args.budget,
        "use_adaptive_budget": False,
        "verbose": True,
        "n_jobs": args.n_jobs,
        "use_multi_fidelity": bool(args.multi_fidelity),
        "fidelity_warm_start": warm_start,
    }
    if args.multi_fidelity:
        if fidelity_levels is not None:
            scenario_kwargs["fidelity_levels"] = fidelity_levels
        if args.fidelity_promotion_ratio is not None:
            scenario_kwargs["fidelity_promotion_ratio"] = args.fidelity_promotion_ratio
        if args.fidelity_min_configs is not None:
            scenario_kwargs["fidelity_min_configs"] = args.fidelity_min_configs

    scenario = Scenario(**scenario_kwargs)

    instances = [Instance(name=args.problem, n_var=args.n_var, kwargs={})]
    seeds = [args.seed + i for i in range(args.n_seeds)]

    def _mean(scores: list[float]) -> float:
        return float(np.mean(scores))

    task = TuningTask(
        name=f"tune_{args.problem}_{args.algorithm}",
        param_space=param_space,
        instances=instances,
        seeds=seeds,
        aggregator=_mean,
        budget_per_run=budget_per_run,
        maximize=True,
    )

    tuner = RacingTuner(task=task, scenario=scenario, seed=args.seed)
    eval_fn = make_evaluator(
        args.problem,
        args.n_var,
        args.n_obj,
        args.algorithm,
        args.pop_size,
        args.ref_point,
        warm_start and args.multi_fidelity,
    )
    best_config, history = tuner.run(eval_fn)

    best_score = max(trial.score for trial in history)
    print("\nBest config found (higher is better):")
    for k, v in best_config.items():
        print(f"  {k}: {v}")
    print(f"Score: {best_score:.6f}")


if __name__ == "__main__":
    main()
