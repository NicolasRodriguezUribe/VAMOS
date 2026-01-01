from __future__ import annotations

from typing import Protocol

import numpy as np


class ProblemProtocol(Protocol):
    n_var: int
    n_obj: int
    xl: float | int | np.ndarray
    xu: float | int | np.ndarray
    encoding: str

    def evaluate(self, X: np.ndarray, out: dict) -> None: ...


class MixedProblemProtocol(ProblemProtocol, Protocol):
    mixed_spec: dict


__all__ = ["ProblemProtocol", "MixedProblemProtocol"]
