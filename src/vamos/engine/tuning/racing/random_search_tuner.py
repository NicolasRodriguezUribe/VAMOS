from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .sampler import Sampler, UniformSampler
from .tuning_task import EvalContext, TuningTask


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


@dataclass
class TrialResult:
    trial_id: int
    config: dict[str, Any]
    score: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RandomSearchTuner:
    """
    Simple random-search tuner for VAMOS.
    """

    task: TuningTask
    max_trials: int
    seed: int = 0
    rng: np.random.Generator = field(init=False)
    sampler: Sampler | None = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        if self.sampler is None:
            self.sampler = UniformSampler(self.task.param_space)

    def run(
        self,
        eval_fn: Callable[[dict[str, Any], EvalContext], float],
        verbose: bool = True,
    ) -> tuple[dict[str, Any], list[TrialResult]]:
        history: list[TrialResult] = []
        best_score: float | None = None
        best_config: dict[str, Any] | None = None
        assert self.sampler is not None

        for trial_id in range(self.max_trials):
            config = self.sampler.sample(self.rng)

            try:
                self.task.param_space.validate(config)
            except ValueError as e:
                if verbose:
                    _logger().info("[trial %s] invalid config skipped: %s", trial_id, e)
                continue

            score = self.task.eval_config(config, eval_fn)

            if verbose:
                direction = "max" if self.task.maximize else "min"
                _logger().info("[trial %s] %s score=%.6f", trial_id, direction, score)

            trial = TrialResult(trial_id=trial_id, config=config, score=score)
            history.append(trial)

            if best_score is None:
                best_score = score
                best_config = config
            else:
                if self.task.maximize and score > best_score:
                    best_score = score
                    best_config = config
                elif not self.task.maximize and score < best_score:
                    best_score = score
                    best_config = config

        if best_config is None:
            raise RuntimeError("Tuner finished without a valid configuration")

        if verbose:
            _logger().info("[tuner] Best score=%.6f", best_score)

        return best_config, history


__all__ = ["RandomSearchTuner", "TrialResult"]
