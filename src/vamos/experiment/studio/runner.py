from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from vamos.experiment.optimize import OptimizationResult


class DynamicsCallback:
    def __init__(self) -> None:
        self.history: list[np.ndarray] = []

    def __call__(self, algorithm: Any) -> None:
        try:
            if hasattr(algorithm, "pop") and algorithm.pop is not None:
                F = algorithm.pop.get("F")
                if F is not None:
                    self.history.append(np.array(F))
        except Exception:
            pass


def run_focused_optimization(
    problem: str,
    reference_point: np.ndarray,
    algo: str,
    budget: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    from vamos.engine.algorithm.config import NSGAIIConfig
    from vamos.experiment.optimize import OptimizeConfig, _run_config
    from vamos.foundation.problem.registry import make_problem_selection

    selection = make_problem_selection(problem)
    cfg = (
        NSGAIIConfig.builder()
        .pop_size(40)
        .offspring_size(40)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", pressure=2)
        .result_mode("population")
        .build()
    )
    run_cfg = OptimizeConfig(
        problem=selection.instantiate(),
        algorithm="nsgaii",
        algorithm_config=cfg,
        termination=("n_eval", budget),
        seed=0,
        engine="numpy",
    )
    result = _run_config(run_cfg)
    F = result.F
    if F is None:
        raise RuntimeError("Focused optimization returned no objectives.")

    from vamos.ux.analysis.mcdm import reference_point_scores

    scores = reference_point_scores(F, reference_point).scores
    order = np.argsort(scores)
    X = result.X[order] if result.X is not None else None
    return F[order], X


def run_with_history(
    problem_name: str,
    config: dict[str, Any],
    budget: int,
) -> tuple[OptimizationResult, list[np.ndarray]]:
    from vamos.engine.algorithm.config import NSGAIIConfig
    from vamos.engine.algorithm.config.types import AlgorithmConfigProtocol
    from vamos.experiment.optimize import _build_algorithm_config
    from vamos.experiment.optimize import OptimizeConfig, _run_config
    from vamos.engine.config.variation import normalize_operator_tuple
    from vamos.foundation.problem.registry import make_problem_selection

    selection = make_problem_selection(problem_name)
    problem = selection.instantiate()

    algo_name = config.get("algorithm", "nsgaii")
    algo_cfg_raw = config.get("algorithm_config", {})

    def _coerce_algo_config(cfg: Any) -> AlgorithmConfigProtocol:
        if not cfg:
            return _build_algorithm_config(
                algo_name,
                pop_size=100,
                n_var=getattr(problem, "n_var", None),
                n_obj=getattr(problem, "n_obj", None),
            )
        if isinstance(cfg, AlgorithmConfigProtocol):
            return cfg
        if not isinstance(cfg, dict):
            raise TypeError("algorithm_config must be a config object (e.g., NSGAIIConfig.default(...)).")

        if algo_name.lower() != "nsgaii":
            raise TypeError("algorithm_config dict coercion is only supported for NSGA-II; pass a config object instead.")

        builder = NSGAIIConfig.builder()
        if cfg.get("pop_size") is not None:
            builder.pop_size(int(cfg["pop_size"]))
        if cfg.get("offspring_size") is not None:
            builder.offspring_size(int(cfg["offspring_size"]))
        if cfg.get("crossover") is not None:
            normalized = normalize_operator_tuple(cfg["crossover"])
            if normalized is None:
                raise ValueError("Invalid crossover spec.")
            method, params = normalized
            builder.crossover(method, params=params)
        if cfg.get("mutation") is not None:
            normalized = normalize_operator_tuple(cfg["mutation"])
            if normalized is None:
                raise ValueError("Invalid mutation spec.")
            method, params = normalized
            builder.mutation(method, params=params)
        if cfg.get("selection") is not None:
            normalized = normalize_operator_tuple(cfg["selection"])
            if normalized is None:
                raise ValueError("Invalid selection spec.")
            method, params = normalized
            builder.selection(method, **params)
        if cfg.get("repair") is not None:
            normalized = normalize_operator_tuple(cfg["repair"])
            if normalized is None:
                raise ValueError("Invalid repair spec.")
            method, params = normalized
            builder.repair(method, **params)
        if cfg.get("mutation_prob_factor") is not None:
            builder.mutation_prob_factor(float(cfg["mutation_prob_factor"]))
        if cfg.get("result_mode") is not None:
            builder.result_mode(str(cfg["result_mode"]))
        if cfg.get("archive_type") is not None:
            builder.archive_type(str(cfg["archive_type"]))
        if cfg.get("archive") is not None:
            archive_cfg = cfg["archive"]
            if not isinstance(archive_cfg, dict):
                raise TypeError("archive must be a dict.")
            size = int(archive_cfg.get("size", 0))
            kwargs = {k: v for k, v in archive_cfg.items() if k != "size" and v is not None}
            builder.archive(size, **kwargs)
        if cfg.get("constraint_mode") is not None:
            builder.constraint_mode(str(cfg["constraint_mode"]))
        if cfg.get("track_genealogy") is not None:
            builder.track_genealogy(bool(cfg["track_genealogy"]))
        if "adaptive_operator_selection" in cfg:
            aos_raw = cfg.get("adaptive_operator_selection")
            if aos_raw is not None and not isinstance(aos_raw, dict):
                raise TypeError("adaptive_operator_selection must be a dict or null.")
            builder.adaptive_operator_selection(aos_raw)

        return builder.build()

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

    result = _run_config(run_cfg)
    return result, callback.history


__all__ = ["run_focused_optimization", "run_with_history"]
