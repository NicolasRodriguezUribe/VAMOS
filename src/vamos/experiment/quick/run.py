from __future__ import annotations

from typing import Any

from vamos.experiment.optimize import OptimizeConfig, OptimizationResult, optimize
from vamos.foundation.problem.types import ProblemProtocol


def run_optimization(
    *,
    problem: ProblemProtocol,
    algorithm: str,
    algorithm_config: dict[str, Any],
    max_evaluations: int,
    seed: int,
    engine: str,
) -> OptimizationResult:
    """Execute optimization with a fully resolved config."""
    return optimize(
        OptimizeConfig(
            problem=problem,
            algorithm=algorithm,
            algorithm_config=algorithm_config,
            termination=("n_eval", max_evaluations),
            seed=seed,
            engine=engine,
        )
    )
