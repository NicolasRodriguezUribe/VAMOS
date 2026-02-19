from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from vamos.foundation.encoding import EncodingLike


class ProblemProtocol(Protocol):
    n_var: int
    n_obj: int
    n_constraints: int
    n_constr: int
    xl: float | int | np.ndarray
    xu: float | int | np.ndarray
    encoding: EncodingLike

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None: ...


class MixedProblemProtocol(ProblemProtocol, Protocol):
    mixed_spec: dict[str, Any]


__all__ = ["ProblemProtocol", "MixedProblemProtocol"]
