from __future__ import annotations

import csv
import logging
from pathlib import Path
from shutil import copy2
from typing import Protocol
from collections.abc import Iterable, Sequence

from vamos.foundation.core.experiment_config import ExperimentConfig
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

    def mirror_artifacts(self, result: StudyResult) -> None:
        """
        Mirror per-run artifacts (like FUN.csv) to a consolidated directory.
        """
        ...


class CSVPersister:
    """
    Persists study results to a CSV file and optionally mirrors artifacts.
    """

    def __init__(self, mirror_roots: Sequence[str | Path] | None = None):
        roots = mirror_roots or ()
        self._mirror_roots = tuple(Path(root) for root in roots)

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

    def mirror_artifacts(self, result: StudyResult) -> None:
        if not self._mirror_roots:
            return

        metrics = result.metrics
        output_dir = metrics.get("output_dir")
        if not output_dir:
            return

        src_dir = Path(output_dir).resolve()
        base_root = Path(ExperimentConfig().output_root).resolve()
        relative = None
        try:
            relative = src_dir.relative_to(base_root)
        except ValueError:
            relative = None

        for root in self._mirror_roots:
            target_root = Path(root).resolve()
            if relative is not None:
                dst = target_root / relative
            else:
                dst = target_root / src_dir.name
            if dst.resolve() == src_dir:
                continue
            dst.mkdir(parents=True, exist_ok=True)
            for name in ("FUN.csv", "ARCHIVE_FUN.csv", "time.txt", "metadata.json"):
                src_file = src_dir / name
                if src_file.exists():
                    copy2(src_file, dst / name)
