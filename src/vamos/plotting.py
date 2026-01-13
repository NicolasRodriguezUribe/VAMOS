"""
Public plotting namespace.

Re-exports plotting helpers from vamos.ux.plotting.
"""

from __future__ import annotations

from typing import Any

from .ux import plotting as _plotting

__all__ = _plotting.__all__


def __getattr__(name: str) -> Any:
    if name in __all__:
        return getattr(_plotting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
