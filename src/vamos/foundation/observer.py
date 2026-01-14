from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass
class RunContext:
    """
    Encapsulates the static context of an optimization run.
    Passed to on_start events.
    """

    problem: Any  # Problem instance
    algorithm: Any  # Algorithm instance (if available during start)
    config: Any  # ExperimentConfig or internal run config
    selection: Any = None  # ProblemSelection (optional)
    algorithm_name: str = "unknown"
    engine_name: str = "unknown"


@runtime_checkable
class Observer(Protocol):
    """
    Observer interface for the Event Bus.
    Reacts to lifecycle events of the optimization process.
    """

    def on_start(self, ctx: RunContext) -> None:
        """Called effectively once at the beginning of the run."""
        ...

    def on_generation(
        self,
        generation: int,
        F: np.ndarray | None = None,
        X: np.ndarray | None = None,
        stats: dict[str, Any] | None = None,
    ) -> None:
        """Called at every generation (or step) of the algorithm."""
        ...

    def on_end(
        self,
        final_F: np.ndarray | None = None,
        final_stats: dict[str, Any] | None = None,
    ) -> None:
        """Called once at the end of the run."""
        ...
