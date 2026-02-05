"""
Unified API for VAMOS optimization.

This module provides a single, powerful entry point that consolidates
problem-based runs, study-style configuration, and auto-parameter defaults
into one flexible function.
"""

from __future__ import annotations

import logging
import numbers
from collections.abc import Mapping
from typing import cast, overload

from vamos.engine.algorithm.config.types import AlgorithmConfigProtocol
from vamos.experiment.optimization_result import OptimizationResult
from vamos.experiment.optimize import _OptimizeConfig, _run_config, _build_algorithm_config
from vamos.foundation.eval import EvaluationBackend
from vamos.foundation.encoding import normalize_encoding
from vamos.foundation.logging import configure_vamos_logging
from vamos.foundation.core.experiment_config import resolve_engine
from vamos.foundation.problem.types import ProblemProtocol
from vamos.experiment.auto import _resolve_problem, _select_algorithm, _compute_pop_size, _compute_max_evaluations


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


_ALLOWED_EVAL_STRATEGIES = {"serial", "multiprocessing", "dask"}


def _coerce_int(name: str, value: object, *, min_value: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer.")
    parsed = int(value)
    if min_value is not None and parsed < min_value:
        if min_value == 1:
            raise ValueError(f"{name} must be a positive integer.")
        raise ValueError(f"{name} must be >= {min_value}.")
    return parsed


def _resolve_problem_label(problem: object, instance: ProblemProtocol) -> str:
    if isinstance(problem, str):
        return problem
    for attr in ("name", "key", "label"):
        label = getattr(instance, attr, None)
        if isinstance(label, str) and label.strip():
            return label
    return instance.__class__.__name__


def _extract_max_evaluations(termination: object) -> int | None:
    if not isinstance(termination, tuple) or len(termination) != 2:
        return None
    kind, value = cast(tuple[object, object], termination)
    if isinstance(kind, str) and kind == "max_evaluations":
        return _coerce_int("termination max_evaluations", value, min_value=1)
    return None


@overload
def optimize(
    problem: str | ProblemProtocol,
    *,
    algorithm: str = "auto",
    max_evaluations: int | None = None,
    pop_size: int | None = None,
    engine: str | None = None,
    seed: int = 42,
    verbose: bool = False,
    n_var: int | None = None,
    n_obj: int | None = None,
    problem_kwargs: Mapping[str, object] | None = None,
    algorithm_config: AlgorithmConfigProtocol | None = None,
    termination: tuple[str, object] | None = None,
    eval_strategy: EvaluationBackend | str | None = None,
    live_viz: object | None = None,
    checkpoint: object | None = None,
) -> OptimizationResult: ...


@overload
def optimize(
    problem: str | ProblemProtocol,
    *,
    algorithm: str = "auto",
    max_evaluations: int | None = None,
    pop_size: int | None = None,
    engine: str | None = None,
    seed: list[int] | tuple[int, ...],
    verbose: bool = False,
    n_var: int | None = None,
    n_obj: int | None = None,
    problem_kwargs: Mapping[str, object] | None = None,
    algorithm_config: AlgorithmConfigProtocol | None = None,
    termination: tuple[str, object] | None = None,
    eval_strategy: EvaluationBackend | str | None = None,
    live_viz: object | None = None,
    checkpoint: object | None = None,
) -> list[OptimizationResult]: ...


def optimize(
    problem: str | ProblemProtocol,
    *,
    algorithm: str = "auto",
    max_evaluations: int | None = None,
    pop_size: int | None = None,
    engine: str | None = None,
    seed: int | list[int] | tuple[int, ...] = 42,
    verbose: bool = False,
    n_var: int | None = None,
    n_obj: int | None = None,
    problem_kwargs: Mapping[str, object] | None = None,
    algorithm_config: AlgorithmConfigProtocol | None = None,
    termination: tuple[str, object] | None = None,
    eval_strategy: EvaluationBackend | str | None = None,
    live_viz: object | None = None,
    checkpoint: object | None = None,
) -> OptimizationResult | list[OptimizationResult]:
    """
    Unified entry point for VAMOS optimization.

    This function consolidates multiple APIs into a single powerful interface:
    - Accepts problem names (strings) or instances
    - Supports AutoML with algorithm="auto"
    - Handles multi-run studies with seed=[0,1,2,...]
    - Prefer optimize(...) for all runs (explicit options are available).

    Args:
        problem: Problem name (e.g., "zdt1") or problem instance.
        algorithm: Algorithm name or "auto" for automatic selection.
        max_evaluations: Maximum function evaluations. Auto-determined if None.
        pop_size: Population size. Auto-determined if None.
        engine: Backend engine ("numpy", "numba", "moocore", "jax").
        seed: Random seed or list of seeds for multi-run mode.
        verbose: Print progress information.
        n_var: Override problem dimension when using a string problem key.
        n_obj: Override objective count when using a string problem key.
        problem_kwargs: Extra kwargs forwarded to problem instantiation for string problems.
        algorithm_config: Optional algorithm config object.
        termination: Optional termination tuple; overrides max_evaluations if provided.
        eval_strategy: Evaluation backend name or instance (e.g., "serial", "dask").
        live_viz: Optional live visualization callback.
        checkpoint: Optional checkpoint payload to warm-start compatible algorithms.

    Returns:
        OptimizationResult for single seed, or list[OptimizationResult] for multiple seeds.

    Examples:
        # AutoML mode - zero config
        >>> result = vamos.optimize("zdt1")

        # Specify algorithm
        >>> result = vamos.optimize("zdt1", algorithm="moead", max_evaluations=5000)

        # Multi-seed study
        >>> results = vamos.optimize("zdt1", seed=[0, 1, 2, 3, 4])
    """
    # Multi-seed mode
    if isinstance(seed, (list, tuple)):
        if checkpoint is not None:
            raise ValueError("checkpoint is only supported for single-seed runs.")
        return [
            _run_single(
                problem,
                algorithm,
                max_evaluations,
                pop_size,
                engine,
                single_seed,
                verbose,
                n_var,
                n_obj,
                problem_kwargs,
                algorithm_config,
                termination,
                eval_strategy,
                live_viz,
                checkpoint,
            )
            for single_seed in seed
        ]

    # Single run
    return _run_single(
        problem,
        algorithm,
        max_evaluations,
        pop_size,
        engine,
        seed,
        verbose,
        n_var,
        n_obj,
        problem_kwargs,
        algorithm_config,
        termination,
        eval_strategy,
        live_viz,
        checkpoint,
    )


def _run_single(
    problem: str | ProblemProtocol,
    algorithm: str,
    max_evaluations: int | None,
    pop_size: int | None,
    engine: str | None,
    seed: int,
    verbose: bool,
    n_var: int | None,
    n_obj: int | None,
    problem_kwargs: Mapping[str, object] | None,
    algorithm_config: AlgorithmConfigProtocol | None,
    termination: tuple[str, object] | None,
    eval_strategy: EvaluationBackend | str | None,
    live_viz: object | None,
    checkpoint: object | None,
) -> OptimizationResult:
    """Execute a single optimization run."""
    if problem_kwargs is not None and not isinstance(problem_kwargs, Mapping):
        raise TypeError("problem_kwargs must be a mapping of keyword arguments.")
    if n_var is not None:
        n_var = _coerce_int("n_var", n_var, min_value=1)
    if n_obj is not None:
        n_obj = _coerce_int("n_obj", n_obj, min_value=1)
    if pop_size is not None:
        pop_size = _coerce_int("pop_size", pop_size, min_value=1)
    if max_evaluations is not None:
        max_evaluations = _coerce_int("max_evaluations", max_evaluations, min_value=1)
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise TypeError("seed must be an integer.")
    if termination is not None:
        if not isinstance(termination, tuple) or len(termination) != 2:
            raise TypeError("termination must be a (kind, value) tuple.")
        term_kind, term_value = termination
        if not isinstance(term_kind, str):
            raise TypeError("termination kind must be a string.")
        if term_kind == "hv" and not isinstance(term_value, Mapping):
            raise TypeError("termination=('hv', ...) requires a mapping payload.")
    if isinstance(eval_strategy, str):
        eval_key = eval_strategy.lower()
        if eval_key not in _ALLOWED_EVAL_STRATEGIES:
            choices = ", ".join(sorted(_ALLOWED_EVAL_STRATEGIES))
            raise ValueError(f"eval_strategy must be one of: {choices}.")

    if verbose:
        configure_vamos_logging()

    algorithm_was_auto = algorithm == "auto"

    # Resolve problem
    problem_instance = _resolve_problem(problem, n_var=n_var, n_obj=n_obj, problem_kwargs=problem_kwargs)

    # Extract metadata
    n_var = getattr(problem_instance, "n_var", 10)
    n_obj = getattr(problem_instance, "n_obj", 2)
    encoding = normalize_encoding(getattr(problem_instance, "encoding", "real"))

    if algorithm_config is not None and algorithm == "auto":
        raise ValueError("algorithm_config requires an explicit algorithm name (not algorithm='auto').")

    # Auto-select algorithm if needed
    if algorithm == "auto":
        algorithm = _select_algorithm(n_obj, encoding)
        if verbose:
            _logger().info("[vamos] Auto-selected algorithm: %s", algorithm)

    # Auto-determine hyperparameters if not specified
    effective_pop_size = pop_size if pop_size else _compute_pop_size(n_var, n_obj)
    term_max_evaluations = _extract_max_evaluations(termination) if termination is not None else None
    if termination is not None and max_evaluations is not None:
        if term_max_evaluations is None:
            raise ValueError("max_evaluations can only be combined with termination=('max_evaluations', max_evaluations).")
        if term_max_evaluations != max_evaluations:
            raise ValueError(f"max_evaluations={max_evaluations} conflicts with termination={termination}.")
    if termination is not None:
        effective_max_evaluations = term_max_evaluations
        effective_termination = termination
    else:
        effective_max_evaluations = max_evaluations if max_evaluations is not None else _compute_max_evaluations(n_var, n_obj)
        effective_termination = ("max_evaluations", effective_max_evaluations)
    effective_engine = resolve_engine(engine, algorithm=algorithm)

    if verbose:
        _logger().info("[vamos] Problem: n_var=%s, n_obj=%s, encoding=%s", n_var, n_obj, encoding)
        _logger().info(
            "[vamos] Config: %s, pop_size=%s, max_evaluations=%s",
            algorithm,
            effective_pop_size,
            effective_max_evaluations,
        )

    algo_cfg: AlgorithmConfigProtocol
    if algorithm_config is None:
        algo_cfg = _build_algorithm_config(
            algorithm,
            pop_size=effective_pop_size,
            n_var=n_var,
            n_obj=n_obj,
        )
    else:
        if not isinstance(algorithm_config, AlgorithmConfigProtocol):
            raise TypeError(
                "algorithm_config must be a config object (e.g., NSGAIIConfig.default(...), or GenericAlgorithmConfig for plugin algorithms)."
            )
        cfg_dict = dict(algorithm_config.to_dict())
        if pop_size is not None:
            cfg_pop_size = cfg_dict.get("pop_size")
            if cfg_pop_size is None:
                raise TypeError(
                    "pop_size cannot be provided unless algorithm_config defines 'pop_size'; set it on algorithm_config instead."
                )
            if isinstance(cfg_pop_size, bool):
                raise TypeError("algorithm_config 'pop_size' must be an int.")
            if isinstance(cfg_pop_size, int):
                cfg_pop_size_int = cfg_pop_size
            elif isinstance(cfg_pop_size, str):
                try:
                    cfg_pop_size_int = int(cfg_pop_size)
                except ValueError as exc:
                    raise TypeError("algorithm_config 'pop_size' must be an int.") from exc
            else:
                raise TypeError("algorithm_config 'pop_size' must be an int.")
            if int(pop_size) != cfg_pop_size_int:
                raise ValueError(
                    f"Conflicting pop_size: pop_size={pop_size} but algorithm_config.pop_size={cfg_pop_size_int}. "
                    "Set pop_size on algorithm_config (single source of truth)."
                )
        algo_cfg = algorithm_config

    algo_cfg_dict = dict(algo_cfg.to_dict())
    resolved_pop_size = algo_cfg_dict.get("pop_size")
    if resolved_pop_size is None and algorithm_config is None:
        resolved_pop_size = effective_pop_size

    config = _OptimizeConfig(
        problem=problem_instance,
        algorithm=algorithm,
        algorithm_config=algo_cfg,
        termination=effective_termination,
        seed=seed,
        engine=effective_engine,
        eval_strategy=eval_strategy,
        live_viz=live_viz,
        checkpoint=checkpoint,
    )
    result = _run_config(config)
    problem_label = _resolve_problem_label(problem, problem_instance)
    resolved_config = {
        "problem": problem_label,
        "algorithm": algorithm,
        "engine": effective_engine,
        "pop_size": resolved_pop_size,
        "max_evaluations": effective_max_evaluations,
        "seed": seed,
        "n_var": n_var,
        "n_obj": n_obj,
        "encoding": encoding,
    }
    pop_size_source = "config" if algorithm_config is not None else ("explicit" if pop_size is not None else "auto")
    max_evaluations_source = "explicit" if termination is not None or max_evaluations is not None else "auto"
    result.meta["resolved_config"] = resolved_config
    result.meta["default_sources"] = {
        "algorithm": "auto" if algorithm_was_auto else "explicit",
        "pop_size": pop_size_source,
        "max_evaluations": max_evaluations_source,
        "engine": "auto" if engine is None else "explicit",
        "algorithm_config": "auto" if algorithm_config is None else "explicit",
    }
    return result


__all__ = ["optimize"]
