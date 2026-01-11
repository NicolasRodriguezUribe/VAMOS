"""
Interoperability with the pymoo library (https://pymoo.org).
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from vamos.foundation.problem.types import ProblemProtocol


class PymooProblem(Protocol):
    """Protocol for pymoo problem objects."""

    n_var: int
    n_obj: int
    n_constr: int
    xl: np.ndarray
    xu: np.ndarray

    def evaluate(self, X: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any) -> Any: ...


class PymooProblemAdapter(ProblemProtocol):
    """
    Adapts a pymoo problem to the VAMOS ProblemProtocol.

    Example:
        >>> from pymoo.problems import get_problem
        >>> from vamos_contrib.interop.pymoo import PymooProblemAdapter
        >>> pymoo_prob = get_problem("zdt1")
        >>> prob = PymooProblemAdapter(pymoo_prob)
        >>> vamos.optimize(prob, ...)
    """

    def __init__(self, problem: PymooProblem) -> None:
        self._problem = problem
        self.n_var = problem.n_var
        self.n_obj = problem.n_obj
        self.n_constr = problem.n_constr
        self.xl = problem.xl
        self.xu = problem.xu
        # Assume real encoding unless pymoo says otherwise (pymoo usually has explicit types,
        # but standardized metadata like 'encoding' is not guaranteed, assume continuous).
        self.encoding = "real"

    def evaluate(self, X: np.ndarray, out: dict[str, Any]) -> None:
        """Evaluate complying with VAMOS in-place protocol."""
        # Pymoo also uses in-place 'out' dict, so direct delegation usually works!
        self._problem.evaluate(X, out)


__all__ = ["PymooProblemAdapter"]
