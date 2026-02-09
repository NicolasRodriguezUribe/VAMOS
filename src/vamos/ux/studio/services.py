from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


from vamos.engine.algorithm.config import GenericAlgorithmConfig, NSGAIIConfig
from vamos.engine.algorithm.config.defaults import build_default_algorithm_config
from vamos.engine.algorithm.config.types import AlgorithmConfigProtocol
from vamos.engine.algorithm.registry import get_algorithms_registry, resolve_algorithm
from vamos.engine.config.variation import normalize_operator_tuple
from vamos.exceptions import InvalidAlgorithmError
from vamos.foundation.eval import EvaluationBackend
from vamos.foundation.eval.backends import resolve_eval_strategy
from vamos.foundation.kernel.registry import resolve_kernel
from vamos.foundation.problem.registry import make_problem_selection
from vamos.foundation.problem.types import ProblemProtocol
from vamos.ux.analysis.mcdm import reference_point_scores

from vamos.ux.studio.data import build_fronts, load_runs_from_study
from vamos.ux.studio.dm import build_decision_view

if TYPE_CHECKING:
    from pathlib import Path
    from vamos.ux.studio.data import FrontRecord, RunRecord
    from vamos.ux.studio.dm import DecisionView


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
            _logger().debug("Failed to capture population snapshot in DynamicsCallback", exc_info=True)


def load_studio_data(study_dir: Path) -> tuple[list[RunRecord], list[FrontRecord]]:
    runs = load_runs_from_study(study_dir)
    fronts = build_fronts(runs)
    return runs, fronts


def build_decision_views(
    fronts: list[FrontRecord],
    weights: np.ndarray,
    reference_point: np.ndarray | None,
    method: str,
) -> list[DecisionView]:
    views = []
    for front in fronts:
        view = build_decision_view(front, weights=weights, reference_point=reference_point, methods=[method, "weighted_sum", "knee"])
        views.append(view)
    return views


def _with_result_mode(cfg_data: AlgorithmConfigProtocol, result_mode: str) -> AlgorithmConfigProtocol:
    fields_map = getattr(cfg_data, "__dataclass_fields__", None)
    if not isinstance(fields_map, dict):
        return cfg_data
    if "result_mode" not in fields_map:
        return cfg_data
    return cast(AlgorithmConfigProtocol, replace(cast(Any, cfg_data), result_mode=result_mode))


def _build_algorithm_config(
    algorithm: str,
    *,
    pop_size: int | None,
    n_var: int | None,
    n_obj: int | None,
) -> AlgorithmConfigProtocol:
    algorithm = algorithm.lower()
    result_mode = "population"

    default_cfg = build_default_algorithm_config(
        algorithm,
        pop_size=pop_size,
        n_var=n_var,
        n_obj=n_obj,
    )
    if default_cfg is not None:
        return _with_result_mode(default_cfg, result_mode)

    registry = get_algorithms_registry()
    if algorithm in registry:
        base: dict[str, object] = {}
        if pop_size is not None:
            base["pop_size"] = pop_size
        if n_var is not None:
            base["n_var"] = n_var
        if n_obj is not None:
            base["n_obj"] = n_obj
        return GenericAlgorithmConfig(base)

    available = sorted(registry.keys())
    raise InvalidAlgorithmError(algorithm, available=available)


def _run_algorithm(
    problem: ProblemProtocol,
    *,
    algorithm: str,
    algorithm_config: AlgorithmConfigProtocol,
    termination: tuple[str, object],
    seed: int,
    engine: str,
    eval_strategy: EvaluationBackend | str | None = None,
    live_viz: object | None = None,
) -> dict[str, Any]:
    cfg_dict = dict(algorithm_config.to_dict())
    if "engine" in cfg_dict:
        raise ValueError("engine must be configured via run arguments, not algorithm_config.")
    kernel = resolve_kernel(engine)

    if eval_strategy is not None:
        backend = resolve_eval_strategy(eval_strategy) if isinstance(eval_strategy, str) else eval_strategy
    else:
        backend_name = str(cfg_dict.get("eval_strategy", "serial"))
        backend = resolve_eval_strategy(backend_name)

    algo_ctor = resolve_algorithm(algorithm)
    algorithm_instance = algo_ctor(cfg_dict, kernel)
    run_fn = getattr(algorithm_instance, "run")
    result = run_fn(
        problem=problem,
        termination=termination,
        seed=seed,
        eval_strategy=backend,
        live_viz=live_viz,
    )
    return dict(result)


def run_focused_optimization(
    problem: str,
    reference_point: np.ndarray,
    algo: str,
    budget: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    selection = make_problem_selection(problem)
    instance = selection.instantiate()
    algo_name = algo or "nsgaii"
    algo_cfg = _build_algorithm_config(
        algo_name,
        pop_size=40,
        n_var=getattr(instance, "n_var", None),
        n_obj=getattr(instance, "n_obj", None),
    )
    result = _run_algorithm(
        instance,
        algorithm=algo_name,
        algorithm_config=algo_cfg,
        termination=("max_evaluations", budget),
        seed=0,
        engine="numpy",
    )
    F = result.get("F")
    if F is None:
        raise RuntimeError("Focused optimization returned no objectives.")

    scores = reference_point_scores(np.asarray(F), reference_point).scores
    order = np.argsort(scores)
    X = None
    if result.get("X") is not None:
        X = np.asarray(result["X"])[order]
    return np.asarray(F)[order], X


def run_with_history(
    problem_name: str,
    config: dict[str, Any],
    budget: int,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    selection = make_problem_selection(problem_name)
    problem = selection.instantiate()

    algo_name = str(config.get("algorithm", "nsgaii"))
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

    result = _run_algorithm(
        problem,
        algorithm=algo_name,
        algorithm_config=algo_cfg,
        termination=("max_evaluations", budget),
        seed=int(config.get("seed", 0)),
        engine=str(config.get("engine", "numpy")),
        live_viz=callback,
    )
    return result, callback.history


__all__ = [
    "DynamicsCallback",
    "load_studio_data",
    "build_decision_views",
    "run_focused_optimization",
    "run_with_history",
]
