"""
Persistence helpers for VAMOS run artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_population(
    output_dir: str | Path,
    F: np.ndarray,
    archive: Optional[dict] = None,
    *,
    X: Optional[np.ndarray] = None,
    G: Optional[np.ndarray] = None,
) -> dict:
    """
    Save population (F and optionally X/G) and optional archive to CSV files.

    Returns:
        dict: artifact names keyed by a short label for metadata wiring.
    """
    out = {}
    output_dir = ensure_dir(output_dir)

    fun_path = output_dir / "FUN.csv"
    np.savetxt(fun_path, F, delimiter=",")
    out["fun"] = fun_path.name

    if X is not None:
        x_path = output_dir / "X.csv"
        np.savetxt(x_path, X, delimiter=",")
        out["x"] = x_path.name

    if G is not None:
        g_path = output_dir / "G.csv"
        np.savetxt(g_path, G, delimiter=",")
        out["g"] = g_path.name

    if archive is not None:
        archive_F = archive.get("F")
        archive_X = archive.get("X")
        archive_G = archive.get("G")
        if archive_F is not None:
            archive_fun = output_dir / "ARCHIVE_FUN.csv"
            np.savetxt(archive_fun, archive_F, delimiter=",")
            out["archive_fun"] = archive_fun.name
        if archive_X is not None:
            archive_x = output_dir / "ARCHIVE_X.csv"
            np.savetxt(archive_x, archive_X, delimiter=",")
            out["archive_x"] = archive_x.name
        if archive_G is not None:
            archive_g = output_dir / "ARCHIVE_G.csv"
            np.savetxt(archive_g, archive_G, delimiter=",")
            out["archive_g"] = archive_g.name

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
