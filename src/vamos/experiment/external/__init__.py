"""
External baseline adapters and dispatch.
"""

from typing import Any

from .registry import (
    ExternalAlgorithmAdapter,
    EXTERNAL_ALGORITHM_RUNNERS,
    resolve_external_algorithm,
    run_external,
    _get_external_algorithm_adapters,
)


def __getattr__(name: str) -> Any:
    if name == "EXTERNAL_ALGORITHM_ADAPTERS":
        return _get_external_algorithm_adapters()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))


__all__ = [
    "ExternalAlgorithmAdapter",
    "EXTERNAL_ALGORITHM_ADAPTERS",
    "EXTERNAL_ALGORITHM_RUNNERS",
    "resolve_external_algorithm",
    "run_external",
]
