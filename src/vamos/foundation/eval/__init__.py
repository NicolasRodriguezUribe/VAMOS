from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass
class EvaluationResult:
    """Container for objective/constraint evaluation outputs."""

    F: NDArray[np.float64]
    G: NDArray[np.float64] | None = None


class EvaluationBackend(Protocol):
    """Protocol for evaluation backends."""

    def evaluate(self, X: NDArray[np.generic], problem: Any) -> EvaluationResult: ...

    def close(self) -> None:  # pragma: no cover - optional for async backends
        """Clean up any resources (executors, pools)."""
        return None


__all__ = ["EvaluationBackend", "EvaluationResult"]
