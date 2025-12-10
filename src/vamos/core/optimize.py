from __future__ import annotations

import inspect
import warnings
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Mapping, Tuple

from vamos.algorithm.registry import resolve_algorithm, ALGORITHMS
from vamos.kernel.numpy_backend import NumPyKernel
from vamos.kernel.registry import resolve_kernel
from vamos.problem.types import ProblemProtocol
from vamos.eval.backends import resolve_eval_backend


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
    eval_backend: Any = None  # name or backend instance
    live_viz: Any = None


def _normalize_cfg(cfg: Any) -> dict[str, Any]:
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    if is_dataclass(cfg):
        return asdict(cfg)
    return dict(cfg)


def _infer_algorithm_name(cfg_dict: Mapping[str, Any]) -> str:
    """
    Legacy heuristic to infer algorithm name when not provided.
    Kept for backward compatibility; emits deprecation warnings upstream.
    """
    if "survival" in cfg_dict or cfg_dict.get("algorithm", "").lower() == "nsgaii":
        return "nsgaii"
    if "neighbor_size" in cfg_dict or "aggregation" in cfg_dict:
        return "moead"
    if "reference_point" in cfg_dict and "selection" in cfg_dict:
        return "smsemoa"
    if "reference_directions" in cfg_dict:
        return "nsga3"
    if "archive_size" in cfg_dict and "k_neighbors" in cfg_dict:
        return "spea2"
    if "indicator" in cfg_dict:
        return "ibea"
    if any(k in cfg_dict for k in ("inertia", "c1", "c2", "vmax_fraction")):
        return "smpso"
    return ""


def _available_algorithms() -> str:
    return ", ".join(sorted(ALGORITHMS.keys()))


def optimize(
    config_or_problem: OptimizeConfig | ProblemProtocol,
    *legacy_args,
    **legacy_kwargs,
) -> OptimizationResult:
    """
    Run a single optimization for the provided problem/config pair.

    Preferred usage: pass an OptimizeConfig with explicit algorithm and kernel.
    Legacy usage (optimize(problem, config, termination, seed, ...)) is still
    supported but emits a DeprecationWarning.
    """
    if isinstance(config_or_problem, OptimizeConfig):
        cfg = config_or_problem
    else:
        legacy_config = legacy_args[0] if len(legacy_args) >= 1 else legacy_kwargs.get("config")
        legacy_termination = legacy_args[1] if len(legacy_args) >= 2 else legacy_kwargs.get("termination")
        legacy_seed = legacy_args[2] if len(legacy_args) >= 3 else legacy_kwargs.get("seed")
        if legacy_config is None or legacy_termination is None or legacy_seed is None:
            raise TypeError(
                "optimize() legacy signature requires problem, config, termination, seed. "
                "Prefer OptimizeConfig(problem=..., algorithm=..., ...)."
            )
        warnings.warn(
            "optimize(problem, config, termination, seed, ...) is deprecated; "
            "use OptimizeConfig with explicit algorithm instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        problem = config_or_problem
        config = legacy_config
        termination = legacy_termination
        seed = legacy_seed
        engine = legacy_kwargs.get("engine", "numpy")
        eval_backend = legacy_kwargs.get("eval_backend", None)
        live_viz = legacy_kwargs.get("live_viz", None)
        algorithm = legacy_kwargs.get("algorithm", None)
        cfg = OptimizeConfig(
            problem=problem,
            algorithm=algorithm or "",
            algorithm_config=config,
            termination=termination,
            seed=seed,
            engine=engine,
            eval_backend=eval_backend,
            live_viz=live_viz,
        )

    cfg_dict = _normalize_cfg(cfg.algorithm_config)
    algorithm_name = (cfg.algorithm or "").lower()
    if not algorithm_name:
        algorithm_name = _infer_algorithm_name(cfg_dict)
        if not algorithm_name:
            raise ValueError(
                "algorithm must be specified in OptimizeConfig. "
                f"Available algorithms: {_available_algorithms()}"
            )
        warnings.warn(
            f"Inferred algorithm '{algorithm_name}' from config; please set OptimizeConfig.algorithm explicitly.",
            DeprecationWarning,
            stacklevel=2,
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
