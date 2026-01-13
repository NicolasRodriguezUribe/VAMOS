from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.eval import EvaluationBackend
from vamos.foundation.eval.backends import resolve_eval_strategy


class Evaluator(Protocol):
    """Protocol for evaluator configuration in experiment runners."""

    def resolve(self, config: ExperimentConfig) -> EvaluationBackend: ...


@dataclass(frozen=True)
class EvaluatorSpec:
    backend: str | EvaluationBackend | None = None
    n_workers: int | None = None
    chunk_size: int | None = None
    dask_address: str | None = None

    def resolve(self, config: ExperimentConfig) -> EvaluationBackend:
        backend = self.backend
        if backend is None:
            backend = getattr(config, "eval_strategy", "serial")
        if isinstance(backend, str):
            n_workers = self.n_workers if self.n_workers is not None else getattr(config, "n_workers", None)
            return resolve_eval_strategy(
                backend,
                n_workers=n_workers,
                chunk_size=self.chunk_size,
                dask_address=self.dask_address,
            )
        return backend


class TerminationCriterion(Protocol):
    """Protocol for termination configuration in experiment runners."""

    def resolve(self, config: ExperimentConfig) -> tuple[str, Any]: ...


@dataclass(frozen=True)
class EvaluationsTermination:
    max_evaluations: int | None = None

    def resolve(self, config: ExperimentConfig) -> tuple[str, Any]:
        limit = self.max_evaluations if self.max_evaluations is not None else config.max_evaluations
        return ("n_eval", int(limit))


@dataclass(frozen=True)
class HVTermination:
    hv_stop_config: dict[str, Any]
    max_evaluations: int | None = None

    def resolve(self, config: ExperimentConfig) -> tuple[str, Any]:
        payload = dict(self.hv_stop_config)
        payload["max_evaluations"] = int(self.max_evaluations) if self.max_evaluations is not None else config.max_evaluations
        return ("hv", payload)


def resolve_evaluator(
    evaluator: Evaluator | EvaluationBackend | str | None,
    config: ExperimentConfig,
) -> EvaluationBackend:
    if evaluator is None:
        name = getattr(config, "eval_strategy", "serial")
        n_workers = getattr(config, "n_workers", None)
        return resolve_eval_strategy(name, n_workers=n_workers)
    if isinstance(evaluator, str):
        n_workers = getattr(config, "n_workers", None)
        return resolve_eval_strategy(evaluator, n_workers=n_workers)
    if hasattr(evaluator, "resolve"):
        return evaluator.resolve(config)
    return evaluator


def resolve_termination(
    termination: TerminationCriterion | tuple[str, Any] | None,
    config: ExperimentConfig,
    *,
    hv_stop_config: dict[str, Any] | None = None,
    algorithm_name: str | None = None,
) -> tuple[str, Any]:
    if termination is None:
        if hv_stop_config is not None and algorithm_name == "nsgaii":
            return HVTermination(hv_stop_config).resolve(config)
        return EvaluationsTermination().resolve(config)
    if isinstance(termination, tuple):
        return termination
    return termination.resolve(config)


__all__ = [
    "Evaluator",
    "EvaluatorSpec",
    "TerminationCriterion",
    "EvaluationsTermination",
    "HVTermination",
    "resolve_evaluator",
    "resolve_termination",
]
