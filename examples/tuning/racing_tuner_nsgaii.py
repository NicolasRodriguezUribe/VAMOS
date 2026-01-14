"""
Racing tuner (irace-style) for NSGA-II hyperparameters on ZDT1.

Defines a ParamSpace, wraps NSGA-II evaluations, and reports the best config.

Usage:
    python examples/racing_tuner_nsgaii.py
"""

from __future__ import annotations

import numpy as np

from vamos import optimize
from vamos.engine.tuning.api import Int, Instance, ParamSpace, RacingTuner, Real, Scenario, TuningTask
from vamos.foundation.problem.zdt1 import ZDT1Problem as ZDT1
from vamos.algorithms import NSGAIIConfig


def build_param_space() -> ParamSpace:
    """Hyperparameters to tune for NSGA-II."""
    return ParamSpace(
        params={
            "pop_size": Int("pop_size", 40, 120, log=True),
            "crossover_prob": Real("crossover_prob", 0.7, 0.95),
            "mutation_prob": Real("mutation_prob", 0.01, 0.2),
            "mutation_eta": Real("mutation_eta", 5.0, 30.0),
            "selection_pressure": Int("selection_pressure", 2, 4),
        }
    )


def make_algo_config(assignment: dict[str, float]) -> NSGAIIConfig:
    """Translate sampled hyperparameters into an NSGA-II config."""
    return (
        NSGAIIConfig.builder()
        .pop_size(int(assignment["pop_size"]))
        .offspring_size(int(assignment["pop_size"]))
        .crossover("sbx", prob=float(assignment["crossover_prob"]), eta=20.0)
        .mutation(
            "pm",
            prob=float(assignment["mutation_prob"]),
            eta=float(assignment["mutation_eta"]),
        )
        .selection("tournament", pressure=int(assignment["selection_pressure"]))
        .build()
    )


def evaluate_config(config: dict[str, float], ctx) -> float:
    """
    Run NSGA-II with the proposed hyperparameters and return a scalar score.
    Lower is better here: we minimize the average of f1 + f2 across the front.
    """
    algo_cfg = make_algo_config(config)
    result = optimize(
        ZDT1(n_var=ctx.instance.n_var),
        algorithm="nsgaii",
        algorithm_config=algo_cfg,
        termination=("n_eval", ctx.budget),
        seed=ctx.seed,
        engine="numpy",
    )
    F = result.F
    if F is None or len(F) == 0:
        return float("inf")
    return float(np.mean(F[:, 0] + F[:, 1]))


def main() -> None:
    param_space = build_param_space()
    task = TuningTask(
        name="nsgaii_zdt1",
        param_space=param_space,
        instances=[Instance(name="zdt1_30", n_var=30)],
        seeds=[0, 1],
        budget_per_run=1500,
        maximize=False,  # we minimize the score returned by evaluate_config
    )
    scenario = Scenario(
        max_experiments=12,  # total config x instance x seed evaluations
        min_survivors=2,
        elimination_fraction=0.5,
        start_instances=1,
        verbose=True,
    )

    tuner = RacingTuner(task=task, scenario=scenario, seed=0, max_initial_configs=6)
    best_config, history = tuner.run(evaluate_config)

    best_score = min(trial.score for trial in history)
    print("\nBest config found (lower is better):")
    for k, v in best_config.items():
        print(f"  {k}: {v}")
    print(f"Score: {best_score:.6f}")

    tuned_cfg = make_algo_config(best_config)
    print("\nReusable NSGA-II config:")
    print(tuned_cfg)


if __name__ == "__main__":
    main()
