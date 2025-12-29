from __future__ import annotations

import csv

from vamos.adaptation.aos.controller import SummaryRow, TraceRow
from vamos.adaptation.aos.logging import SUMMARY_HEADER, TRACE_HEADER, write_aos_summary, write_aos_trace


def test_write_aos_trace_headers_and_row(tmp_path) -> None:
    path = tmp_path / "aos_trace.csv"
    rows = [
        TraceRow(
            step=1,
            mating_id=0,
            op_id="op1",
            op_name="Operator 1",
            reward=0.4,
            reward_survival=0.2,
            reward_nd_insertions=0.6,
            reward_hv_delta=0.0,
            batch_size=4,
        )
    ]
    write_aos_trace(path, rows)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        assert header == TRACE_HEADER
        data_row = next(reader)
        assert data_row[0] == "1"
        assert data_row[2] == "op1"


def test_write_aos_summary_headers_and_row(tmp_path) -> None:
    path = tmp_path / "aos_summary.csv"
    rows = [
        SummaryRow(
            op_id="op1",
            op_name="Operator 1",
            pulls=5,
            mean_reward=0.3,
            total_reward=1.5,
            usage_fraction=0.5,
        )
    ]
    write_aos_summary(path, rows)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        assert header == SUMMARY_HEADER
        data_row = next(reader)
        assert data_row[0] == "op1"
        assert data_row[2] == "5"
