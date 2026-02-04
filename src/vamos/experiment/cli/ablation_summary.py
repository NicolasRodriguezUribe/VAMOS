"""
Summary output helpers for ablation runs.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def write_summary_csv(
    results: Sequence[Any],
    variant_names: Sequence[str],
    path: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for result, variant in zip(results, variant_names):
        row = result.to_row()
        row["variant"] = variant
        rows.append(row)

    if not rows:
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


__all__ = ["write_summary_csv"]
