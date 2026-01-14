from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass
class EvaluationResult:
    """Container for objective/constraint evaluation outputs."""

    F: np.ndarray
    G: np.ndarray | None = None


class EvaluationBackend(Protocol):
    """Protocol for evaluation backends."""

    def evaluate(self, X: np.ndarray, problem: Any) -> EvaluationResult: ...

    def close(self) -> None:  # pragma: no cover - optional for async backends
        """Clean up any resources (executors, pools)."""
        return None


__all__ = ["EvaluationBackend", "EvaluationResult"]
