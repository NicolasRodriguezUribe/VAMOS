from __future__ import annotations

from vamos.experiment.optimize import OptimizeConfig, OptimizationResult, optimize_config
from vamos.engine.algorithm.config import AlgorithmConfigProtocol
from vamos.foundation.problem.types import ProblemProtocol


def _run_optimization(
    *,
    problem: ProblemProtocol,
    algorithm: str,
    algorithm_config: AlgorithmConfigProtocol,
    max_evaluations: int,
    seed: int,
    engine: str,
) -> OptimizationResult:
    """Execute optimization with a fully resolved config."""
    return optimize_config(
        OptimizeConfig(
            problem=problem,
            algorithm=algorithm,
            algorithm_config=algorithm_config,
            termination=("n_eval", max_evaluations),
            seed=seed,
            engine=engine,
        )
    )
