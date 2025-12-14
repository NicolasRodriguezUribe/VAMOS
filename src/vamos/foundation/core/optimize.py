from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Callable, Mapping, Tuple, Optional

from vamos.engine.algorithm.registry import resolve_algorithm, ALGORITHMS, AlgorithmLike
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.kernel.registry import resolve_kernel
from vamos.foundation.problem.types import ProblemProtocol
from vamos.foundation.eval.backends import resolve_eval_backend, EvaluationBackend


class OptimizationResult:
    """Simple container returned by optimize()."""

    def __init__(self, payload: Mapping[str, Any]):
        self.F = payload.get("F")
        self.X = payload.get("X")
        self.data = dict(payload)

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
    return ", ".join(sorted(ALGORITHMS.keys()))


def optimize(
    config: OptimizeConfig,
) -> OptimizationResult:
    """
    Run a single optimization for the provided problem/config pair.
    """
    if not isinstance(config, OptimizeConfig):
        raise TypeError("optimize() expects an OptimizeConfig instance.")
    cfg = config

    cfg_dict = _normalize_cfg(cfg.algorithm_config)
    algorithm_name = (cfg.algorithm or "").lower()
    if not algorithm_name:
        raise ValueError(
            "OptimizeConfig.algorithm must be specified. "
            f"Available algorithms: {_available_algorithms()}"
        )
    if algorithm_name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm '{algorithm_name}'. Available: {_available_algorithms()}")

    effective_engine = cfg.engine or cfg_dict.get("engine", "numpy")
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


__all__ = ["OptimizeConfig", "optimize", "OptimizationResult"]
