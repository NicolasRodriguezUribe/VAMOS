from __future__ import annotations

import logging
from dataclasses import dataclass, fields, replace
from typing import Any, cast

from vamos.engine.algorithm.registry import resolve_algorithm, get_algorithms_registry
from vamos.foundation.kernel.registry import resolve_kernel
from vamos.foundation.problem.types import ProblemProtocol
from vamos.foundation.eval import EvaluationBackend
from vamos.foundation.eval.backends import resolve_eval_strategy
from vamos.engine.algorithm.config.types import AlgorithmConfigProtocol
from vamos.exceptions import InvalidAlgorithmError


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


from .optimization_result import OptimizationResult


@dataclass
class OptimizeConfig:
    """
    Canonical configuration for a single optimization run.
    """

    problem: ProblemProtocol
    algorithm: str
    algorithm_config: AlgorithmConfigProtocol
    termination: tuple[str, Any]
    seed: int
    engine: str = "numpy"
    eval_strategy: EvaluationBackend | str | None = None  # name or backend instance
    live_viz: Any = None


def _normalize_cfg(cfg: AlgorithmConfigProtocol) -> dict[str, object]:
    return dict(cfg.to_dict())


def _with_result_mode(cfg_data: Any, result_mode: str) -> Any:
    try:
        allowed = {field.name for field in fields(cfg_data)}
    except TypeError:
        return cfg_data
    if "result_mode" not in allowed:
        return cfg_data
    return replace(cfg_data, result_mode=result_mode)


def _run_config(
    config: OptimizeConfig,
    *,
    engine: str | None = None,
) -> OptimizationResult:
    """
    Run a single optimization for the provided problem/config pair.

    Args:
        config: OptimizeConfig with problem, algorithm, and settings
        engine: Override backend engine ('numpy', 'numba', 'moocore').
                If provided, overrides config.engine.

    Returns:
        OptimizationResult with Pareto front and helper methods

    Examples:
        # Standard usage
        result = _run_config(config)

        # Override engine at call time
        result = _run_config(config, engine="numba")
    """
    if not isinstance(config, OptimizeConfig):
        raise TypeError("_run_config() expects an OptimizeConfig instance.")
    cfg = config

    cfg_dict = _normalize_cfg(cfg.algorithm_config)
    if "engine" in cfg_dict:
        raise ValueError("engine must be configured via OptimizeConfig.engine (or _run_config(engine=...)), not algorithm_config.")
    algorithm_raw = cfg.algorithm or ""
    algorithm_name = algorithm_raw.lower()
    available = sorted(get_algorithms_registry().keys())
    if not algorithm_name:
        raise ValueError(f"OptimizeConfig.algorithm must be specified. Available: {', '.join(available)}.")
    registry = get_algorithms_registry()
    if algorithm_name not in registry:
        raise InvalidAlgorithmError(algorithm_raw, available=available)

    effective_engine = engine or cfg.engine
    kernel = resolve_kernel(effective_engine)

    if cfg.eval_strategy is not None:
        backend = resolve_eval_strategy(cfg.eval_strategy) if isinstance(cfg.eval_strategy, str) else cfg.eval_strategy
    else:
        backend_name = str(cfg_dict.get("eval_strategy", "serial"))
        backend = resolve_eval_strategy(backend_name)

    algo_ctor = resolve_algorithm(algorithm_name)
    algorithm = algo_ctor(cfg_dict, kernel)

    run_fn = getattr(algorithm, "run")
    kwargs = {
        "problem": cfg.problem,
        "termination": cfg.termination,
        "seed": cfg.seed,
        "eval_strategy": backend,
        "live_viz": cfg.live_viz,
    }
    result = run_fn(**kwargs)
    return OptimizationResult(
        result,
        meta={
            "algorithm": algorithm_name,
            "engine": effective_engine,
            "seed": cfg.seed,
            "termination": cfg.termination,
        },
    )


def _build_algorithm_config(
    algorithm: str,
    *,
    pop_size: int | None,
    n_var: int | None,
    n_obj: int | None,
) -> AlgorithmConfigProtocol:
    algorithm = algorithm.lower()
    result_mode = "population"

    from vamos.engine.algorithm.config import GenericAlgorithmConfig
    from vamos.engine.algorithm.config.defaults import build_default_algorithm_config

    default_cfg = build_default_algorithm_config(
        algorithm,
        pop_size=pop_size,
        n_var=n_var,
        n_obj=n_obj,
    )
    if default_cfg is not None:
        return cast(AlgorithmConfigProtocol, _with_result_mode(default_cfg, result_mode))

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


__all__ = ["OptimizeConfig", "OptimizationResult"]
