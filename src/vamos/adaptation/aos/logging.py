"""
CSV writers for AOS trace and summary data.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable


TRACE_HEADER = [
    "step",
    "mating_id",
    "op_id",
    "op_name",
    "reward",
    "reward_survival",
    "reward_nd_insertions",
    "reward_hv_delta",
    "batch_size",
]

SUMMARY_HEADER = [
    "op_id",
    "op_name",
    "pulls",
    "mean_reward",
    "total_reward",
    "usage_fraction",
]


def _row_to_dict(row: Any) -> dict[str, Any]:
    if is_dataclass(row):
        return asdict(row)
    if isinstance(row, dict):
        return dict(row)
    raise TypeError("Rows must be dataclasses or dictionaries.")


def write_aos_trace(path: str | Path, rows: Iterable[Any]) -> None:
    """
    Write AOS trace rows to CSV with a fixed header.
    """
    output_path = Path(path)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TRACE_HEADER)
        writer.writeheader()
        for row in rows:
            data = _row_to_dict(row)
            writer.writerow({key: data.get(key) for key in TRACE_HEADER})


def write_aos_summary(path: str | Path, rows: Iterable[Any]) -> None:
    """
    Write AOS summary rows to CSV with a fixed header.
    """
    output_path = Path(path)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_HEADER)
        writer.writeheader()
        for row in rows:
            data = _row_to_dict(row)
            writer.writerow({key: data.get(key) for key in SUMMARY_HEADER})


__all__ = [
    "TRACE_HEADER",
    "SUMMARY_HEADER",
    "write_aos_trace",
    "write_aos_summary",
]
