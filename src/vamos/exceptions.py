"""
Public exceptions namespace.

Re-exports the canonical exception classes from vamos.foundation.exceptions.
"""

from __future__ import annotations

from .foundation import exceptions as _exceptions

__all__ = _exceptions.__all__


def __getattr__(name: str):
    if name in __all__:
        return getattr(_exceptions, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
