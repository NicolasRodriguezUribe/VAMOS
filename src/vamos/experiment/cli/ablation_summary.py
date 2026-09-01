"""
Summary output helpers for ablation runs.
"""

from __future__ import annotations

import csv
from pathlib import Path

from vamos.experiment.ablation import AblationResult


def write_summary_csv(
    result: AblationResult,
    path: Path,
) -> None:
    rows = list(result.summary_rows())

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
