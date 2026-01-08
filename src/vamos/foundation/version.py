"""
Version helpers for VAMOS.
"""

from __future__ import annotations


# Managed by python-semantic-release
__version__ = "0.1.0"


def get_version() -> str:
    return __version__


def __getattr__(name: str):
    if name == "__version__":
        return get_version()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({"__version__", "get_version"} | set(globals()))


__all__ = ["__version__", "get_version"]
