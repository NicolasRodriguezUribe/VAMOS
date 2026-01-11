"""
Public exceptions namespace.

Re-exports the canonical exception classes from vamos.foundation.exceptions.
"""

from __future__ import annotations

from .foundation.exceptions import (
    BackendNotAvailableError,
    BoundsError,
    ConfigurationError,
    ConstraintViolationError,
    ConvergenceError,
    DataError,
    DependencyError,
    EvaluationError,
    InvalidAlgorithmError,
    InvalidEngineError,
    InvalidOperatorError,
    InvalidProblemError,
    InvalidResultsError,
    MissingConfigError,
    OptimizationError,
    ProblemDimensionError,
    ProblemError,
    ResultsNotFoundError,
    VAMOSError,
)

__all__ = [
    "VAMOSError",
    "ConfigurationError",
    "InvalidAlgorithmError",
    "InvalidEngineError",
    "InvalidOperatorError",
    "MissingConfigError",
    "ProblemError",
    "InvalidProblemError",
    "ProblemDimensionError",
    "BoundsError",
    "OptimizationError",
    "ConvergenceError",
    "EvaluationError",
    "ConstraintViolationError",
    "DataError",
    "ResultsNotFoundError",
    "InvalidResultsError",
    "DependencyError",
    "BackendNotAvailableError",
]


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
