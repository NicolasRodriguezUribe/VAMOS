"""
Version helpers for VAMOS.
"""

from __future__ import annotations

from importlib import metadata as importlib_metadata
from typing import TYPE_CHECKING

_VERSION: str | None = None

if TYPE_CHECKING:
    __version__: str


def get_version() -> str:
    global _VERSION
    if _VERSION is not None:
        return _VERSION
    try:  # pragma: no cover - fallback when metadata is unavailable
        _VERSION = importlib_metadata.version("vamos")
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        _VERSION = "0.0.0+unknown"
    return _VERSION


def __getattr__(name: str):
    if name == "__version__":
        return get_version()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({"__version__", "get_version"} | set(globals()))


__all__ = ["__version__", "get_version"]
