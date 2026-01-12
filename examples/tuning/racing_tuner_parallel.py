"""
Parallel Racing Tuner Example

This example demonstrates how to use the RacingTuner to optimize NSGA-II parameters
on the ZDT1 problem using parallel evaluation (n_jobs=4).

Usage:
    python examples/racing_tuner_parallel.py
"""

from __future__ import annotations

import numpy as np

from vamos.api import OptimizeConfig, optimize
from vamos.engine.api import NSGAIIConfig
from vamos.engine.tuning.api import Instance, Int, ParamSpace, RacingTuner, Real, Scenario, TuningTask
from vamos.foundation.problems_registry import ZDT1


def evaluate_config(config: dict, ctx) -> float:
    """
    Evaluate a configuration.

    Args:
        config: Dictionary of hyperparameter values.
        ctx: Evaluation context containing instance, seed, and budget.

    Returns:
        float: The score to MINIMIZE (lower is better).
    """
    # 1. Build algorithm config from hyperparameters
    algo_cfg = (
        NSGAIIConfig()
        .pop_size(int(config["pop_size"]))
        .offspring_size(int(config["pop_size"]))
        .crossover("sbx", prob=float(config["crossover_prob"]), eta=20.0)
        .mutation("pm", prob=float(config["mutation_prob"]), eta=20.0)
        .selection("tournament", pressure=2)
        .fixed()
    )

    # 2. Instantiate the problem for this specific instance
    # ctx.instance.name might be "zdt1_30"
    problem = ZDT1(n_var=ctx.instance.n_var)

    # 3. Run optimization
    result = optimize(
        OptimizeConfig(
            problem=problem,
            algorithm="nsgaii",
            algorithm_config=algo_cfg,
            termination=("n_eval", ctx.budget),
            seed=ctx.seed,
            engine="numpy",  # Use "numba" for better speed if available
        )
    )

    # 4. Compute metric (Avg(f1 + f2))
    F = result.F
    if F is None or len(F) == 0:
        return float("inf")

    # Simple scalarization for tuning signal
    return float(np.mean(F.sum(axis=1)))


def main():
    # 1. Define Parameter Space
    space = ParamSpace()
    space.add(Int("pop_size", 40, 200, log=True))
    space.add(Real("crossover_prob", 0.7, 1.0))
    space.add(Real("mutation_prob", 0.01, 0.2))

    # 2. Define Tuning Task
    task = TuningTask(
        name="tune_nsgaii_zdt1",
        param_space=space,
        instances=[
            Instance("zdt1_30", n_var=30),  # Can add more instances
        ],
        seeds=[101, 102, 103, 104],  # Seeds to race over
        budget_per_run=2000,  # Evaluations per run
        evaluator=evaluate_config,
        maximize=False,  # Minimize the score
    )

    # 3. Define Racing Scenario
    scenario = Scenario(
        max_experiments=50,  # Total tuning budget (evals of configurations)
        min_survivors=3,  # Keep at least 3 configs
        elimination_fraction=0.3,  # Eliminate 30% worst per stage
        n_jobs=4,  # PARALLEL execution (4 workers)
        verbose=True,
    )

    # 4. Run Tuner
    print(f"Starting parallel tuning with n_jobs={scenario.n_jobs}...")
    tuner = RacingTuner(task, scenario)
    best_config, history = tuner.tune()

    print("\n--- Tuning Complete ---")
    print("Best Configuration:")
    for k, v in best_config.items():
        print(f"  {k}: {v}")

    best_score = min(t.score for t in history)
    print(f"Best Score: {best_score:.6f}")


if __name__ == "__main__":
    main()
