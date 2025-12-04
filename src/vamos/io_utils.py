"""
Persistence helpers for VAMOS run artifacts.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_population(output_dir: str | Path, F: np.ndarray, archive: Optional[dict] = None) -> dict:
    """
    Save population and optional archive to CSV files. Returns artifact map.
    """
    out = {}
    output_dir = ensure_dir(output_dir)
    fun_path = output_dir / "FUN.csv"
    np.savetxt(fun_path, F, delimiter=",")
    out["fun"] = fun_path.name
    if archive is not None and archive.get("F") is not None:
        archive_fun = output_dir / "ARCHIVE_FUN.csv"
        np.savetxt(archive_fun, archive["F"], delimiter=",")
        out["archive_fun"] = archive_fun.name
    return out


def write_metadata(output_dir: str | Path, metadata: dict, resolved_cfg: dict) -> None:
    output_dir = ensure_dir(output_dir)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(resolved_cfg, f, indent=2, sort_keys=True)


def write_timing(output_dir: str | Path, total_time_ms: float) -> None:
    output_dir = ensure_dir(output_dir)
    with (output_dir / "time.txt").open("w", encoding="utf-8") as f:
        f.write(f"{total_time_ms:.2f}\n")


__all__ = ["write_population", "write_metadata", "write_timing", "ensure_dir"]
