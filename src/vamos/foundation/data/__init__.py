"""
Packaged data assets for VAMOS (reference fronts, weight vectors, etc.).
"""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path


def _reference_front_locations() -> list[object]:
    """Return candidate locations containing reference front CSV files."""
    locations: list[object] = []

    # Explicit override for external runs (e.g., remote servers).
    override = os.environ.get("VAMOS_REFERENCE_FRONTS_DIR")
    if override:
        override_path = Path(override)
        if override_path.exists():
            locations.append(override_path)

    # Local-repo fallback: <repo>/data/reference_fronts
    # __file__: .../src/vamos/foundation/data/__init__.py
    repo_data = Path(__file__).resolve().parents[4] / "data" / "reference_fronts"
    if repo_data.exists():
        locations.append(repo_data)

    # Packaged fronts remain the default fallback for installed distributions.
    locations.append(resources.files(__name__) / "reference_fronts")

    return locations


def reference_front_path(name: str) -> Path:
    """
    Return a filesystem path to a packaged reference front CSV by key (e.g., "zdt1").
    """
    target = f"{name.lower()}.csv"
    for base in _reference_front_locations():
        candidates = (
            f"{name}.csv",
            f"{name.lower()}.csv",
            f"{name.upper()}.csv",
        )
        for cand in candidates:
            p = base / cand
            if p.is_file():
                return Path(str(p))

        # Case-insensitive lookup for robustness across filesystems and naming.
        for entry in base.iterdir():
            if entry.name.lower() == target and entry.is_file():
                return Path(str(entry))

    raise ValueError(f"Unknown reference front '{name}'.")


def weight_path(filename: str) -> Path:
    """
    Return a filesystem path to a packaged weight vector file.
    """
    path = resources.files(__name__) / "weights" / filename
    if not path.is_file():
        raise ValueError(f"Unknown weights file '{filename}'.")
    return Path(str(path))


__all__ = ["reference_front_path", "weight_path"]
