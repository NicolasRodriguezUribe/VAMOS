from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vamos.experiment.optimize import OptimizationResult


class DynamicsCallback:
    def __init__(self) -> None:
        self.history: list[np.ndarray] = []

    def __call__(self, algorithm: Any) -> None:
        # Capture current F
        # Logic depends on algorithm structure, assuming standard VAMOS algo with 'pop'
        try:
            if hasattr(algorithm, "pop") and algorithm.pop is not None:
                F = algorithm.pop.get("F")
                if F is not None:
                    # Store as copy
                    self.history.append(np.array(F))
        except Exception:
            pass


def run_focused_optimization(
    problem: str,
    reference_point: np.ndarray,
    algo: str,
    budget: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    # Minimal focused re-run leveraging optimize(); uses reference point to bias ranking via Tchebycheff scores post-hoc.
    from vamos.foundation.problem.registry import make_problem_selection
    from vamos.experiment.optimize import OptimizeConfig, optimize_config
    from vamos.engine.algorithm.config import NSGAIIConfig

    selection = make_problem_selection(problem)
    cfg = (
        NSGAIIConfig()
        .pop_size(40)
        .offspring_size(40)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", pressure=2)
        .result_mode("population")
        .fixed()
    )
    run_cfg = OptimizeConfig(
        problem=selection.instantiate(),
        algorithm="nsgaii",
        algorithm_config=cfg,
        termination=("n_eval", budget),
        seed=0,
        engine="numpy",
    )
    result = optimize_config(run_cfg)
    F = result.F
    if F is None:
        raise RuntimeError("Focused optimization returned no objectives.")
    # Re-rank by distance to reference point
    from vamos.ux.analysis.mcdm import reference_point_scores

    scores = reference_point_scores(F, reference_point).scores
    order = np.argsort(scores)
    X = result.X[order] if result.X is not None else None
    return F[order], X


def run_with_history(
    problem_name: str,
    config: dict[str, Any],
    budget: int,
) -> tuple["OptimizationResult", list[np.ndarray]]:
    from vamos.foundation.problem.registry import make_problem_selection
    from vamos.experiment.optimize import OptimizeConfig, optimize_config
    from vamos.engine.algorithm.config import (
        NSGAIIConfig,
    )
    from vamos.engine.algorithm.config.types import AlgorithmConfigLike

    # Instantiate problem
    selection = make_problem_selection(problem_name)
    problem = selection.instantiate()

    # Adjust config for budget if needed, or assume fixed budget
    # We respect the passed config but might override budget
    # If the config came from resolved_config.json, it might have "termination"

    # Clean config for usage
    algo_name = config.get("algorithm", "nsgaii")
    algo_cfg_raw = config.get("algorithm_config", {})
    if not algo_cfg_raw:
        algo_cfg_raw = NSGAIIConfig().pop_size(100).fixed()

    def _coerce_algo_config(cfg: Any) -> AlgorithmConfigLike:
        if hasattr(cfg, "to_dict"):
            return cfg
        if not isinstance(cfg, dict):
            raise TypeError("algorithm_config must be a config object or dict.")
        return cfg

    algo_cfg = _coerce_algo_config(algo_cfg_raw)

    callback = DynamicsCallback()

    run_cfg = OptimizeConfig(
        problem=problem,
        algorithm=algo_name,
        algorithm_config=algo_cfg,
        termination=("n_eval", budget),
        seed=config.get("seed", 0),
        engine=config.get("engine", "numpy"),
        live_viz=callback,
    )

    result = optimize_config(run_cfg)
    return result, callback.history
