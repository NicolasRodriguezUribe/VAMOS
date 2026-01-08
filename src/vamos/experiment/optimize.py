from __future__ import annotations

import inspect
import logging
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Tuple

from vamos.engine.algorithm.registry import resolve_algorithm, get_algorithms_registry
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.kernel.registry import resolve_kernel
from vamos.foundation.problem.types import ProblemProtocol
from vamos.foundation.eval.backends import resolve_eval_backend, EvaluationBackend


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
    algorithm_config: Any
    termination: Tuple[str, Any]
    seed: int
    engine: str = "numpy"
    eval_backend: EvaluationBackend | str | None = None  # name or backend instance
    live_viz: Any = None


def _normalize_cfg(cfg: Any) -> dict[str, Any]:
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    if is_dataclass(cfg):
        return asdict(cfg)
    return dict(cfg)


def _available_algorithms() -> str:
    return ", ".join(sorted(get_algorithms_registry().keys()))


def optimize(
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
        result = optimize(config)

        # Override engine at call time
        result = optimize(config, engine="numba")
    """
    if not isinstance(config, OptimizeConfig):
        raise TypeError("optimize() expects an OptimizeConfig instance.")
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
    kernel = resolve_kernel(effective_engine) if effective_engine != "numpy" else NumPyKernel()

    if cfg.eval_backend is not None:
        backend = cfg.eval_backend
    else:
        backend_name = cfg_dict["eval_backend"] if isinstance(cfg_dict, dict) and "eval_backend" in cfg_dict else "serial"
        backend = resolve_eval_backend(backend_name)

    algo_ctor = resolve_algorithm(algorithm_name)
    algorithm = algo_ctor(cfg_dict, kernel=kernel)

    run_fn = getattr(algorithm, "run")
    sig = inspect.signature(run_fn)
    kwargs = {"problem": cfg.problem, "termination": cfg.termination, "seed": cfg.seed}
    if "eval_backend" in sig.parameters:
        kwargs["eval_backend"] = backend
    if "live_viz" in sig.parameters:
        kwargs["live_viz"] = cfg.live_viz
    result = run_fn(**kwargs)
    return OptimizationResult(result)


def run_optimization(
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
    Simplified optimization interface - run without creating OptimizeConfig.

    This is a convenience function for quick experiments. For full control,
    use `optimize()` with `OptimizeConfig`.

    Args:
        problem: Problem instance to optimize
        algorithm: Algorithm name ('nsgaii', 'moead', 'spea2', 'smsemoa', 'nsgaiii')
        max_evaluations: Maximum function evaluations
        pop_size: Population size
        engine: Backend engine ('numpy', 'numba', 'moocore')
        seed: Random seed for reproducibility
        **kwargs: Additional algorithm-specific parameters

    Returns:
        OptimizationResult with Pareto front and helper methods

    Examples:
        from vamos.api import run_optimization
        from vamos.foundation.problems_registry import ZDT1

        problem = ZDT1(n_var=30)
        result = run_optimization(problem, "nsgaii", max_evaluations=5000)
        result.summary()
        result.plot()
    """
    from vamos.engine.algorithm.config import (
        NSGAIIConfig,
        MOEADConfig,
        SPEA2Config,
        SMSEMOAConfig,
        NSGAIIIConfig,
    )

    algorithm = algorithm.lower()
    result_mode = kwargs.pop("result_mode", "population")
    n_var = getattr(problem, "n_var", None)

    # Build config based on algorithm
    if algorithm == "nsgaii":
        algo_config = NSGAIIConfig.default(pop_size=pop_size, n_var=n_var, engine=engine)
    elif algorithm == "moead":
        algo_config = MOEADConfig.default(pop_size=pop_size, n_var=n_var, engine=engine)
    elif algorithm == "spea2":
        algo_config = SPEA2Config.default(pop_size=pop_size, n_var=n_var, engine=engine)
    elif algorithm == "smsemoa":
        algo_config = SMSEMOAConfig.default(pop_size=pop_size, n_var=n_var, engine=engine)
    elif algorithm == "nsgaiii":
        n_obj = getattr(problem, "n_obj", 3)
        algo_config = NSGAIIIConfig.default(pop_size=pop_size, n_var=n_var, n_obj=n_obj, engine=engine)
    else:
        from vamos.exceptions import InvalidAlgorithmError

        raise InvalidAlgorithmError(
            algorithm,
            available=["nsgaii", "moead", "spea2", "smsemoa", "nsgaiii"],
        )

    algo_cfg = algo_config.to_dict() if hasattr(algo_config, "to_dict") else dict(algo_config)
    if result_mode is not None:
        algo_cfg["result_mode"] = result_mode
    if kwargs:
        algo_cfg.update(kwargs)

    config = OptimizeConfig(
        problem=problem,
        algorithm=algorithm,
        algorithm_config=algo_cfg,
        termination=("n_eval", max_evaluations),
        seed=seed,
        engine=engine,
    )

    return optimize(config)


__all__ = ["OptimizeConfig", "optimize", "OptimizationResult", "pareto_filter", "run_optimization"]
