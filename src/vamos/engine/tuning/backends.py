from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ._model_backend_availability import available_model_based_backends, require_backend
from ._model_backend_fidelity import (
    default_optuna_study_name,
    eval_config_at_budget,
    resolve_budget_levels,
    resolve_fidelity_slice,
    resolve_seed_bounds,
)
from ._model_backend_runners import run_bohb_native, run_optuna_like, run_smac3
from .racing.random_search_tuner import TrialResult
from .racing.tuning_task import EvalContext, Instance, TuningTask


@dataclass
class ModelBasedTuner:
    """Optional facade for Optuna, BOHB, and SMAC3 tuning backends."""

    task: TuningTask
    max_trials: int
    backend: str = "optuna"
    seed: int = 0
    n_jobs: int = 1
    timeout_seconds: float | None = None
    show_progress_bar: bool = False
    bohb_reduction_factor: int = 3
    budget_levels: list[int] | None = None
    fidelity_min_instance_frac: float = 1.0
    fidelity_min_seed_count: int | None = None
    fidelity_max_seed_count: int | None = None
    fidelity_selection_seed: int | None = None
    optuna_storage_url: str | None = None
    optuna_study_name: str | None = None
    optuna_load_if_exists: bool = True
    optuna_sampler: str = "tpe"
    _fidelity_cache: dict[int, tuple[Sequence[Instance], Sequence[int], dict[str, Any]]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        frac = float(self.fidelity_min_instance_frac)
        if not (0.0 < frac <= 1.0):
            raise ValueError("fidelity_min_instance_frac must be in (0, 1].")
        if self.fidelity_min_seed_count is not None and int(self.fidelity_min_seed_count) <= 0:
            raise ValueError("fidelity_min_seed_count must be > 0 when provided.")
        if self.fidelity_max_seed_count is not None and int(self.fidelity_max_seed_count) <= 0:
            raise ValueError("fidelity_max_seed_count must be > 0 when provided.")
        if (
            self.fidelity_min_seed_count is not None
            and self.fidelity_max_seed_count is not None
            and int(self.fidelity_min_seed_count) > int(self.fidelity_max_seed_count)
        ):
            raise ValueError("fidelity_min_seed_count cannot be greater than fidelity_max_seed_count.")

    def _worst_score(self) -> float:
        return float("-inf") if self.task.maximize else float("inf")

    def _score_to_loss(self, score: float) -> float:
        return float(-score if self.task.maximize else score)

    def _default_optuna_study_name(self) -> str:
        return default_optuna_study_name(self)

    def _resolve_seed_bounds(self) -> tuple[int, int]:
        return resolve_seed_bounds(self)

    def _resolve_fidelity_slice(self, budget: int) -> tuple[Sequence[Instance], Sequence[int], dict[str, Any]]:
        return resolve_fidelity_slice(self, budget)

    def _eval_config_at_budget(
        self,
        config: dict[str, Any],
        eval_fn: Callable[[dict[str, Any], EvalContext], float],
        budget: int,
    ) -> float:
        return eval_config_at_budget(self, config, eval_fn, budget)

    def _resolve_budget_levels(self) -> list[int]:
        return resolve_budget_levels(self)

    def _run_optuna_like(
        self,
        eval_fn: Callable[[dict[str, Any], EvalContext], float],
        bohb_mode: bool,
    ) -> tuple[dict[str, Any], list[TrialResult]]:
        return run_optuna_like(self, eval_fn, bohb_mode=bohb_mode)

    def _run_smac3(self, eval_fn: Callable[[dict[str, Any], EvalContext], float]) -> tuple[dict[str, Any], list[TrialResult]]:
        return run_smac3(self, eval_fn)

    def _run_bohb_native(
        self,
        eval_fn: Callable[[dict[str, Any], EvalContext], float],
    ) -> tuple[dict[str, Any], list[TrialResult]]:
        return run_bohb_native(self, eval_fn)

    def run(
        self,
        eval_fn: Callable[[dict[str, Any], EvalContext], float],
        verbose: bool = True,
    ) -> tuple[dict[str, Any], list[TrialResult]]:
        _ = verbose
        if self.backend == "optuna":
            require_backend("optuna")
            return self._run_optuna_like(eval_fn, bohb_mode=False)
        if self.backend == "bohb_optuna":
            require_backend("bohb_optuna")
            return self._run_optuna_like(eval_fn, bohb_mode=True)
        if self.backend == "smac3":
            require_backend("smac3")
            return self._run_smac3(eval_fn)
        if self.backend == "bohb":
            require_backend("bohb")
            return self._run_bohb_native(eval_fn)
        raise ValueError(f"Unknown backend '{self.backend}'.")


__all__ = ["ModelBasedTuner", "available_model_based_backends"]
