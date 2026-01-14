from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Mapping

import numpy as np

from vamos.foundation.problem.registry import make_problem_selection
from vamos.experiment.unified import optimize
from vamos.foundation.metrics.hypervolume import compute_hypervolume
from vamos.engine.tuning.racing.config_space import AlgorithmConfigSpace
from vamos.engine.tuning.racing.param_space import ParamSpace

from vamos.engine.tuning import (
    RacingTuner,
    Scenario,
    TuningTask,
    Instance,
    EvalContext,
    # Builders
    build_nsgaii_config_space,
    build_moead_config_space,
    build_nsgaiii_config_space,
    build_spea2_config_space,
    build_ibea_config_space,
    build_smpso_config_space,
    build_smsemoa_config_space,
    # Bridge
    config_from_assignment,
)

BUILDERS: dict[str, Callable[[], AlgorithmConfigSpace | ParamSpace]] = {
    "nsgaii": build_nsgaii_config_space,
    "moead": build_moead_config_space,
    "nsgaiii": build_nsgaiii_config_space,
    "spea2": build_spea2_config_space,
    "ibea": build_ibea_config_space,
    "smpso": build_smpso_config_space,
    "smsemoa": build_smsemoa_config_space,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VAMOS Tuning CLI (vamos-tune)")
    parser.add_argument("--problem", type=str, required=True, help="Problem ID (e.g., zdt1)")
    parser.add_argument("--algorithm", type=str, default="nsgaii", choices=list(BUILDERS.keys()), help="Algorithm to tune")
    parser.add_argument("--n-var", type=int, default=30, help="Number of variables")
    parser.add_argument("--n-obj", type=int, default=2, help="Number of objectives")
    parser.add_argument("--budget", type=int, default=5000, help="Max evaluations per run")
    parser.add_argument(
        "--tune-budget",
        type=int,
        default=20000,
        help="Max number of configuration evaluations (config x instance x seed)",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--n-seeds", type=int, default=5, help="Number of seeds to evaluate per config")
    parser.add_argument("--pop-size", type=int, default=100, help="Fixed population size (if not tuning it)")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs (-1 for all cores)")
    parser.add_argument("--ref-point", type=str, default=None, help="Reference point for HV (comma-separated, e.g. '1.1,1.1')")
    return parser.parse_args()


def make_evaluator(
    problem_key: str,
    n_var: int,
    n_obj: int,
    algorithm_name: str,
    fixed_pop_size: int,
    ref_point_str: str | None,
) -> Callable[[Mapping[str, object], EvalContext], float]:
    """
    Creates an evaluation function that runs the algorithm and returns the Hypervolume.
    """
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
        # Default fallback
        ref_point = [1.1] * n_obj

    def eval_fn(config_dict: Mapping[str, object], ctx: EvalContext) -> float:
        try:
            # 1. Prepare Configuration
            # Merge fixed params if missing in tuning config
            start_config: dict[str, object] = dict(config_dict)
            if "pop_size" not in start_config:
                start_config["pop_size"] = fixed_pop_size

            # Use the unified bridge to build the AlgorithmConfig object
            cfg = config_from_assignment(algorithm_name, start_config)

            # 2. Run the algorithm
            selection = make_problem_selection(problem_key, n_var=n_var, n_obj=n_obj)

            result = optimize(
                selection.instantiate(),
                algorithm=algorithm_name,
                algorithm_config=cfg,
                termination=("n_eval", ctx.budget),
                seed=ctx.seed,
                engine="numpy",
            )

            # 3. Compute metric (Hypervolume)
            if result.F is None or len(result.F) == 0:
                return 0.0

            hv = compute_hypervolume(result.F, ref_point)
            return float(hv)

        except Exception as exc:
            # In tuning, we often want to absorb errors and return bad score
            # to keep the racer alive.
            _logger().warning("Eval failed for %s: %s", algorithm_name, exc)
            return 0.0

    return eval_fn


def main() -> None:
    _configure_cli_logging()
    args = parse_args()

    # 1. Define Parameter Space
    builder = BUILDERS[args.algorithm]

    algo_space = builder()
    if isinstance(algo_space, AlgorithmConfigSpace):
        param_space = algo_space.to_param_space()
    else:
        param_space = algo_space

    _logger().info("Tuning %s on %s (Budget: %s)", args.algorithm, args.problem, args.tune_budget)
    _logger().info("Parallel Jobs: %s", args.n_jobs)

    # 2. Setup Scenario and Task
    scenario = Scenario(
        max_experiments=args.tune_budget, initial_budget_per_run=args.budget, use_adaptive_budget=False, verbose=True, n_jobs=args.n_jobs
    )

    # Instance definition
    instances = [Instance(name=args.problem, n_var=args.n_var, kwargs={})]

    # Seeds
    seeds = [args.seed + i for i in range(args.n_seeds)]

    # Aggregator: Mean
    def _mean(scores: list[float]) -> float:
        return float(np.mean(scores))

    aggregator = _mean

    task = TuningTask(
        name=f"tune_{args.problem}_{args.algorithm}",
        param_space=param_space,
        instances=instances,
        seeds=seeds,
        aggregator=aggregator,
        budget_per_run=args.budget,
        maximize=True,  # HV is maximization
    )

    # 3. Run Tuner
    tuner = RacingTuner(task=task, scenario=scenario, seed=args.seed)

    eval_fn = make_evaluator(args.problem, args.n_var, args.n_obj, args.algorithm, args.pop_size, args.ref_point)

    best_config, history = tuner.run(eval_fn)

    _logger().info("--- Tuning Complete ---")
    _logger().info("Best Configuration Found:")
    for k, v in best_config.items():
        _logger().info("  %s: %s", k, v)

    _logger().info("Use these parameters in your scripts or CLI using --config!")


if __name__ == "__main__":
    main()
