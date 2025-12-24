"""
VAMOS exception hierarchy.

Provides user-friendly exceptions with helpful error messages and suggestions.
All VAMOS-specific exceptions inherit from VAMOSError for easy catching.

Example:
    try:
        result = optimize(config)
    except VAMOSError as e:
        print(f"Optimization failed: {e}")
        print(f"Suggestion: {e.suggestion}")
"""

from __future__ import annotations

from typing import Any


class VAMOSError(Exception):
    """
    Base exception for all VAMOS errors.

    Attributes:
        message: Human-readable error description
        suggestion: Optional suggestion for fixing the error
        details: Optional dict with additional context
    """

    def __init__(
        self,
        message: str,
        suggestion: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.suggestion = suggestion
        self.details = details or {}
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the full error message with suggestion."""
        msg = self.message
        if self.suggestion:
            msg += f"\n\nSuggestion: {self.suggestion}"
        return msg


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(VAMOSError):
    """Raised when configuration is invalid or incomplete."""

    pass


class InvalidAlgorithmError(ConfigurationError):
    """Raised when an unknown algorithm is specified."""

    def __init__(self, algorithm: str, available: list[str] | None = None) -> None:
        available = available or [
            "nsgaii",
            "moead",
            "spea2",
            "smsemoa",
            "nsgaiii",
            "ibea",
            "smpso",
        ]
        message = f"Unknown algorithm '{algorithm}'."
        suggestion = f"Available algorithms: {', '.join(available)}"
        super().__init__(message, suggestion, {"algorithm": algorithm, "available": available})


class InvalidEngineError(ConfigurationError):
    """Raised when an unknown backend engine is specified."""

    def __init__(self, engine: str, available: list[str] | None = None) -> None:
        available = available or ["numpy", "numba", "moocore"]
        message = f"Unknown engine '{engine}'."
        suggestion = f"Available engines: {', '.join(available)}. Install extras with: pip install vamos[backends]"
        super().__init__(message, suggestion, {"engine": engine, "available": available})


class InvalidOperatorError(ConfigurationError):
    """Raised when an unknown operator is specified."""

    def __init__(
        self,
        operator_type: str,
        operator_name: str,
        available: list[str] | None = None,
    ) -> None:
        message = f"Unknown {operator_type} operator '{operator_name}'."
        suggestion = f"Available {operator_type} operators: {', '.join(available)}" if available else None
        super().__init__(
            message,
            suggestion,
            {"operator_type": operator_type, "operator_name": operator_name},
        )


class MissingConfigError(ConfigurationError):
    """Raised when required configuration is missing."""

    def __init__(self, field: str, config_class: str | None = None) -> None:
        message = f"Missing required configuration: '{field}'."
        suggestion = f"Add '{field}' to your configuration"
        if config_class:
            suggestion += f" or use {config_class}.default() for sensible defaults"
        super().__init__(message, suggestion, {"field": field})


# =============================================================================
# Problem Errors
# =============================================================================


class ProblemError(VAMOSError):
    """Base class for problem-related errors."""

    pass


class InvalidProblemError(ProblemError):
    """Raised when an unknown problem is specified."""

    def __init__(self, problem: str, available: list[str] | None = None) -> None:
        message = f"Unknown problem '{problem}'."
        if available:
            # Show first few problems as examples
            examples = available[:5]
            suggestion = f"Examples: {', '.join(examples)}. Use available_problem_names() for full list."
        else:
            suggestion = "Use available_problem_names() to see registered problems."
        super().__init__(message, suggestion, {"problem": problem})


class ProblemDimensionError(ProblemError):
    """Raised when problem dimensions are invalid."""

    def __init__(
        self,
        message: str,
        n_var: int | None = None,
        n_obj: int | None = None,
    ) -> None:
        suggestion = "Check problem dimensions: n_var (variables), n_obj (objectives)"
        super().__init__(message, suggestion, {"n_var": n_var, "n_obj": n_obj})


class BoundsError(ProblemError):
    """Raised when bounds are invalid or inconsistent."""

    def __init__(self, message: str) -> None:
        suggestion = "Ensure xl <= xu for all variables and bounds have correct shape"
        super().__init__(message, suggestion)


# =============================================================================
# Runtime Errors
# =============================================================================


class OptimizationError(VAMOSError):
    """Raised when optimization fails during execution."""

    pass


class ConvergenceError(OptimizationError):
    """Raised when optimization fails to converge."""

    def __init__(self, message: str, evaluations: int | None = None) -> None:
        suggestion = "Try increasing max_evaluations or adjusting algorithm parameters"
        super().__init__(message, suggestion, {"evaluations": evaluations})


class EvaluationError(OptimizationError):
    """Raised when objective evaluation fails."""

    def __init__(self, message: str, solution: Any = None) -> None:
        suggestion = "Check your problem's evaluate() function for errors"
        super().__init__(message, suggestion, {"solution": solution})


class ConstraintViolationError(OptimizationError):
    """Raised when constraints cannot be satisfied."""

    def __init__(self, message: str, violations: Any = None) -> None:
        suggestion = "Check constraint definitions or relax constraint bounds"
        super().__init__(message, suggestion, {"violations": violations})


# =============================================================================
# Data/IO Errors
# =============================================================================


class DataError(VAMOSError):
    """Base class for data-related errors."""

    pass


class ResultsNotFoundError(DataError):
    """Raised when expected results files are missing."""

    def __init__(self, path: str) -> None:
        message = f"Results not found at '{path}'."
        suggestion = "Check the path or run optimization first"
        super().__init__(message, suggestion, {"path": path})


class InvalidResultsError(DataError):
    """Raised when results data is invalid or corrupted."""

    def __init__(self, message: str, path: str | None = None) -> None:
        suggestion = "Results may be corrupted. Try re-running the optimization."
        super().__init__(message, suggestion, {"path": path})


# =============================================================================
# Dependency Errors
# =============================================================================


class DependencyError(VAMOSError):
    """Raised when an optional dependency is missing."""

    def __init__(self, package: str, feature: str, install_cmd: str | None = None) -> None:
        message = f"'{package}' is required for {feature} but not installed."
        install_cmd = install_cmd or f"pip install {package}"
        suggestion = f"Install with: {install_cmd}"
        super().__init__(message, suggestion, {"package": package, "feature": feature})


class BackendNotAvailableError(DependencyError):
    """Raised when a requested backend is not installed."""

    def __init__(self, backend: str) -> None:
        feature = f"the '{backend}' backend"
        if backend == "numba":
            install_cmd = "pip install vamos[backends]"
        elif backend == "moocore":
            install_cmd = "pip install vamos[backends]"
        else:
            install_cmd = f"pip install {backend}"
        super().__init__(backend, feature, install_cmd)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Base
    "VAMOSError",
    # Configuration
    "ConfigurationError",
    "InvalidAlgorithmError",
    "InvalidEngineError",
    "InvalidOperatorError",
    "MissingConfigError",
    # Problem
    "ProblemError",
    "InvalidProblemError",
    "ProblemDimensionError",
    "BoundsError",
    # Runtime
    "OptimizationError",
    "ConvergenceError",
    "EvaluationError",
    "ConstraintViolationError",
    # Data/IO
    "DataError",
    "ResultsNotFoundError",
    "InvalidResultsError",
    # Dependencies
    "DependencyError",
    "BackendNotAvailableError",
]
