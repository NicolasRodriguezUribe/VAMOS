"""
Persistence helpers for VAMOS run artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

RESULT_FILES = {
    "fun": "FUN.csv",
    "x": "X.csv",
    "g": "G.csv",
    "archive_fun": "ARCHIVE_FUN.csv",
    "archive_x": "ARCHIVE_X.csv",
    "archive_g": "ARCHIVE_G.csv",
    "metadata": "metadata.json",
    "resolved_config": "resolved_config.json",
    "time": "time.txt",
}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_population(
    output_dir: str | Path,
    F: np.ndarray,
    archive: dict[str, np.ndarray] | None = None,
    *,
    X: np.ndarray | None = None,
    G: np.ndarray | None = None,
) -> dict[str, str]:
    """
    Save population (F and optionally X/G) and optional archive to CSV files.

    Returns:
        dict: artifact names keyed by a short label for metadata wiring.
    """
    out: dict[str, str] = {}
    output_dir = ensure_dir(output_dir)

    fun_path = output_dir / RESULT_FILES["fun"]
    np.savetxt(fun_path, F, delimiter=",")
    out["fun"] = fun_path.name

    if X is not None:
        x_path = output_dir / RESULT_FILES["x"]
        np.savetxt(x_path, X, delimiter=",")
        out["x"] = x_path.name

    if G is not None:
        g_path = output_dir / RESULT_FILES["g"]
        np.savetxt(g_path, G, delimiter=",")
        out["g"] = g_path.name

    if archive is not None:
        archive_F = archive.get("F")
        archive_X = archive.get("X")
        archive_G = archive.get("G")
        if archive_F is not None:
            archive_fun = output_dir / RESULT_FILES["archive_fun"]
            np.savetxt(archive_fun, archive_F, delimiter=",")
            out["archive_fun"] = archive_fun.name
        if archive_X is not None:
            archive_x = output_dir / RESULT_FILES["archive_x"]
            np.savetxt(archive_x, archive_X, delimiter=",")
            out["archive_x"] = archive_x.name
        if archive_G is not None:
            archive_g = output_dir / RESULT_FILES["archive_g"]
            np.savetxt(archive_g, archive_G, delimiter=",")
            out["archive_g"] = archive_g.name

    return out


def write_metadata(output_dir: str | Path, metadata: dict[str, Any], resolved_cfg: dict[str, Any]) -> None:
    output_dir = ensure_dir(output_dir)
    with (output_dir / RESULT_FILES["metadata"]).open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    with (output_dir / RESULT_FILES["resolved_config"]).open("w", encoding="utf-8") as f:
        json.dump(resolved_cfg, f, indent=2, sort_keys=True)


def write_timing(output_dir: str | Path, total_time_ms: float) -> None:
    output_dir = ensure_dir(output_dir)
    with (output_dir / RESULT_FILES["time"]).open("w", encoding="utf-8") as f:
        f.write(f"{total_time_ms:.2f}\n")


__all__ = ["RESULT_FILES", "write_population", "write_metadata", "write_timing", "ensure_dir"]
