"""
External baseline adapters and dispatch.
"""

from .registry import (
    ExternalAlgorithmAdapter,
    EXTERNAL_ALGORITHM_ADAPTERS,
    EXTERNAL_ALGORITHM_RUNNERS,
    resolve_external_algorithm,
    run_external,
)

__all__ = [
    "ExternalAlgorithmAdapter",
    "EXTERNAL_ALGORITHM_ADAPTERS",
    "EXTERNAL_ALGORITHM_RUNNERS",
    "resolve_external_algorithm",
    "run_external",
]
