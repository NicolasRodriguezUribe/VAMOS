"""Small filesystem helpers that do not define an artifact format."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = ["ensure_dir"]
