"""
Version helpers for VAMOS.
"""

from __future__ import annotations

from importlib import metadata as importlib_metadata

try:  # pragma: no cover - fallback when metadata is unavailable
    __version__: str = importlib_metadata.version("vamos")
except importlib_metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
