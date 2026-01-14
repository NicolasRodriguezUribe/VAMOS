"""
Shared helpers for benchmark reporting (IO, stats, formatting).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def import_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Benchmark reporting requires pandas. Install via 'pip install pandas' or the 'analysis'/'examples' extras."
        ) from exc
    return pd


def higher_is_better(metric: str) -> bool:
    m = metric.lower()
    if m in {"igd", "igd+", "igd_plus", "epsilon", "epsilon_additive", "epsilon_mult"}:
        return False
    return True


def format_cell(fmt: str, mean: float, std: float, is_best: bool, marker: str) -> str:
    if fmt.startswith("%"):
        fmt = fmt.lstrip("%")
    cell = f"{mean:{fmt}} +/- {std:{fmt}}"
    if is_best:
        cell = f"\\textbf{{{cell}}}"
    if marker:
        cell = f"{cell}$^{{{marker}}}$"
    return cell


def dump_stats_summary(stats: dict[str, Any], path: Path) -> None:
    serializable = {}
    for metric, payload in stats.items():
        fried = payload.get("friedman")
        serializable[metric] = {
            "friedman": {
                "statistic": getattr(fried, "statistic", None),
                "p_value": getattr(fried, "p_value", None),
            }
            if fried
            else None
        }
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


__all__ = [
    "ensure_dir",
    "import_pandas",
    "higher_is_better",
    "format_cell",
    "dump_stats_summary",
]
