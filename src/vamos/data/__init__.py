"""
Packaged data assets for VAMOS (reference fronts, weight vectors, etc.).
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path


def reference_front_path(name: str) -> Path:
    """
    Return a filesystem path to a packaged reference front CSV by key (e.g., "zdt1").
    """
    path = resources.files(__name__) / "reference_fronts" / f"{name.upper() if name.lower().startswith('zdt') else name}.csv"
    if not path.is_file():
        lower = name.lower()
        alt = resources.files(__name__) / "reference_fronts" / f"{lower}.csv"
        if alt.is_file():
            return Path(alt)
        raise ValueError(f"Unknown reference front '{name}'.")
    return Path(path)


def weight_path(filename: str) -> Path:
    """
    Return a filesystem path to a packaged weight vector CSV.
    """
    path = resources.files(__name__) / "weights" / filename
    if not path.is_file():
        raise ValueError(f"Unknown weights file '{filename}'.")
    return Path(path)


__all__ = ["reference_front_path", "weight_path"]
