from __future__ import annotations

import logging
from dataclasses import dataclass, fields, replace
from typing import Any, Tuple, cast

from vamos.engine.algorithm.registry import resolve_algorithm, get_algorithms_registry
from vamos.foundation.kernel.registry import resolve_kernel
from vamos.foundation.problem.types import ProblemProtocol
from vamos.foundation.eval import EvaluationBackend
from vamos.foundation.eval.backends import resolve_eval_strategy
from vamos.engine.algorithm.config.types import AlgorithmConfigProtocol


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


from vamos.foundation.metrics.pareto import pareto_filter
from .optimization_result import OptimizationResult


@dataclass
class OptimizeConfig:
    """
    Canonical configuration for a single optimization run.
    """

    problem: ProblemProtocol
    algorithm: str
    # Config objects must expose a .to_dict() method.
    algorithm_config: AlgorithmConfigProtocol
    termination: Tuple[str, Any]
    seed: int
    engine: str = "numpy"
    eval_strategy: EvaluationBackend | str | None = None  # name or backend instance
    live_viz: Any = None


def _normalize_cfg(cfg: AlgorithmConfigProtocol) -> dict[str, Any]:
    if not hasattr(cfg, "to_dict"):
        raise TypeError("algorithm_config must provide a .to_dict() method.")
    return dict(cfg.to_dict())


def _apply_overrides(
    cfg_data: Any,
    overrides: dict[str, Any],
    *,
    algo_label: str,
) -> AlgorithmConfigProtocol:
    if not overrides:
        return cast(AlgorithmConfigProtocol, cfg_data)
    allowed = {field.name for field in fields(cfg_data)}
    unknown = sorted(key for key in overrides if key not in allowed)
    if unknown:
        raise ValueError(f"{algo_label} config does not support: {', '.join(unknown)}")
    return cast(AlgorithmConfigProtocol, replace(cfg_data, **{key: overrides[key] for key in overrides}))


def _available_algorithms() -> str:
    return ", ".join(sorted(get_algorithms_registry().keys()))


def optimize_config(
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
        result = optimize_config(config)

        # Override engine at call time
        result = optimize_config(config, engine="numba")
    """
    if not isinstance(config, OptimizeConfig):
        raise TypeError("optimize_config() expects an OptimizeConfig instance.")
    cfg = config

    cfg_dict = _normalize_cfg(cfg.algorithm_config)
    algorithm_name = (cfg.algorithm or "").lower()
    if not algorithm_name:
        raise ValueError(f"OptimizeConfig.algorithm must be specified. Available algorithms: {_available_algorithms()}")
    registry = get_algorithms_registry()
    if algorithm_name not in registry:
        raise ValueError(f"Unknown algorithm '{algorithm_name}'. Available: {_available_algorithms()}")

    # Engine priority: function arg > config > algorithm_config > default
    effective_engine = engine or cfg.engine or cfg_dict.get("engine", "numpy")
    kernel = resolve_kernel(effective_engine)

    if cfg.eval_strategy is not None:
        backend = cfg.eval_strategy
    else:
        backend_name = cfg_dict.get("eval_strategy", "serial")
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
    return OptimizationResult(result)


def _build_algorithm_config(
    algorithm: str,
    *,
    pop_size: int,
    n_var: int | None,
    n_obj: int | None,
    engine: str,
    **kwargs: Any,
) -> AlgorithmConfigProtocol:
    from vamos.engine.algorithm.config import (
        NSGAIIConfig,
        MOEADConfig,
        SPEA2Config,
        SMSEMOAConfig,
        NSGAIIIConfig,
    )

    algorithm = algorithm.lower()
    result_mode = kwargs.pop("result_mode", "population")

    overrides = dict(kwargs)
    if result_mode is not None:
        overrides["result_mode"] = result_mode

    if algorithm == "nsgaii":
        nsgaii_cfg = NSGAIIConfig.default(pop_size=pop_size, n_var=n_var, engine=engine)
        if overrides:
            nsgaii_cfg = NSGAIIConfig.from_dict({**nsgaii_cfg.to_dict(), **overrides})
        return nsgaii_cfg
    if algorithm == "moead":
        moead_cfg = MOEADConfig.default(pop_size=pop_size, n_var=n_var, engine=engine)
        if overrides:
            moead_cfg = MOEADConfig.from_dict({**moead_cfg.to_dict(), **overrides})
        return moead_cfg
    if algorithm == "spea2":
        spea2_cfg = SPEA2Config.default(pop_size=pop_size, n_var=n_var, engine=engine)
        return _apply_overrides(spea2_cfg, overrides, algo_label="SPEA2")
    if algorithm == "smsemoa":
        smsemoa_cfg = SMSEMOAConfig.default(pop_size=pop_size, n_var=n_var, engine=engine)
        return _apply_overrides(smsemoa_cfg, overrides, algo_label="SMS-EMOA")
    if algorithm == "nsgaiii":
        n_obj = n_obj if n_obj is not None else 3
        nsgaiii_cfg = NSGAIIIConfig.default(pop_size=pop_size, n_var=n_var, n_obj=n_obj, engine=engine)
        return _apply_overrides(nsgaiii_cfg, overrides, algo_label="NSGA-III")

    from vamos.exceptions import InvalidAlgorithmError

    raise InvalidAlgorithmError(
        algorithm,
        available=["nsgaii", "moead", "spea2", "smsemoa", "nsgaiii"],
    )


def _optimize_problem(
    problem: ProblemProtocol,
    algorithm: str = "nsgaii",
    *,
    max_evaluations: int = 10000,
    pop_size: int = 100,
    engine: str = "numpy",
    seed: int = 42,
    **kwargs: Any,
) -> OptimizationResult:
    """
    Internal helper to run an optimization with default algorithm configs.
    """
    n_var = getattr(problem, "n_var", None)
    n_obj = getattr(problem, "n_obj", None)
    algo_cfg = _build_algorithm_config(
        algorithm,
        pop_size=pop_size,
        n_var=n_var,
        n_obj=n_obj,
        engine=engine,
        **kwargs,
    )

    config = OptimizeConfig(
        problem=problem,
        algorithm=algorithm,
        algorithm_config=algo_cfg,
        termination=("n_eval", max_evaluations),
        seed=seed,
        engine=engine,
    )

    return optimize_config(config)


__all__ = ["OptimizeConfig", "optimize_config", "OptimizationResult", "pareto_filter"]
