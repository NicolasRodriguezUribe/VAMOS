"""
NSGA-II evolutionary algorithm core.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vamos.engine.algorithm.components.termination import HVTracker
from vamos.foundation.eval.backends import EvaluationBackend
from vamos.foundation.kernel.backend import KernelBackend
from vamos.foundation.problem.types import ProblemProtocol
from vamos.hooks.live_viz import LiveVisualization

from .ask_tell import ask_nsgaii, combine_ids, tell_nsgaii
from .run import notify_generation, run_nsgaii, save_checkpoint
from .setup import initialize_run
from .state import NSGAIIState


class NSGAII:
    """
    Vectorized/SOA-style NSGA-II evolutionary core.
    Individuals are represented as array rows (X, F) without per-object instances.
    """

    def __init__(self, config: dict[str, Any], kernel: KernelBackend) -> None:
        self.cfg = config
        self.kernel = kernel
        self._st: NSGAIIState | None = None

    def run(
        self,
        problem: ProblemProtocol,
        termination: tuple[str, Any],
        seed: int,
        eval_strategy: EvaluationBackend | None = None,
        live_viz: LiveVisualization | None = None,
        checkpoint_dir: str | None = None,
        checkpoint_interval: int = 50,
        checkpoint: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return run_nsgaii(
            self,
            problem,
            termination,
            seed,
            eval_strategy=eval_strategy,
            live_viz=live_viz,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
            checkpoint=checkpoint,
        )

    def _save_checkpoint(self, checkpoint_dir: str, seed: int, generation: int, n_eval: int) -> None:
        save_checkpoint(self, checkpoint_dir, seed, generation, n_eval)

    def _notify_generation(
        self,
        live_cb: LiveVisualization,
        generation: int,
        F: np.ndarray,
        evals: int | None = None,
    ) -> bool:
        return notify_generation(self, live_cb, generation, F, evals=evals)

    def _initialize_run(
        self,
        problem: ProblemProtocol,
        termination: tuple[str, Any],
        seed: int,
        eval_strategy: EvaluationBackend | None,
        live_viz: LiveVisualization | None,
        checkpoint: dict[str, Any] | None = None,
    ) -> tuple[LiveVisualization, EvaluationBackend, int, int, HVTracker]:
        return initialize_run(self, problem, termination, seed, eval_strategy, live_viz, checkpoint=checkpoint)

    def ask(self) -> np.ndarray:
        return ask_nsgaii(self)

    def tell(self, eval_result: Any, pop_size: int) -> bool:
        return tell_nsgaii(self, eval_result, pop_size)

    def _combine_ids(self, st: NSGAIIState) -> np.ndarray | None:
        return combine_ids(st)


__all__ = ["NSGAII"]
