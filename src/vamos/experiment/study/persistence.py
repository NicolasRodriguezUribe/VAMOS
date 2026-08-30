from __future__ import annotations

import csv
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from vamos.experiment.study.types import StudyResult


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class StudyPersister(Protocol):
    """
    Protocol for Components that handle the persistence of StudyResults.
    """

    def save_results(self, results: Iterable[StudyResult], path: str | Path | None = None) -> Path | None:
        """
        Save the aggregated results (e.g. to a CSV file).
        """
        ...


class CSVPersister:
    """Export a derived study summary without copying per-run artifacts."""

    def save_results(self, results: Iterable[StudyResult], path: str | Path | None = None) -> Path | None:
        if path is None:
            return None

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert iterable to list to iterate multiple times if needed,
        # though strictly we only iterate once to build rows
        rows = [res.to_row() for res in results]
        if not rows:
            return path

        seen: dict[str, None] = {}
        for row in rows:
            seen.update(dict.fromkeys(row.keys()))
        fieldnames = sorted(seen.keys())

        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        _logger().info("[Persister] CSV exported to %s", path)
        return path
