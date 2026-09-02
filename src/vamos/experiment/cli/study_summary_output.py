"""Explicit non-authoritative JSON and CSV study-summary publication."""

from __future__ import annotations

import csv
import io
import json
import os
import uuid
from pathlib import Path
from typing import Literal

from vamos.experiment.study.errors import StudyInfrastructureError, StudyOutputCollisionError
from vamos.experiment.study.report_models import StudySummary
from vamos.experiment.study.writing import fsync_directory

SummaryFormat = Literal["json", "csv"]

_TRACE_FIELDS = (
    "derived",
    "canonical_authority",
    "generated_at",
    "root_manifest_sha256",
    "event_head_sequence",
    "event_head_sha256",
)


def write_summary(summary: StudySummary, path: Path, *, output_format: SummaryFormat) -> int:
    """Atomically publish one explicit derived summary without overwriting."""
    destination = path.absolute()
    if os.path.lexists(destination):
        raise _collision(destination)
    payload = _json_bytes(summary) if output_format == "json" else _csv_bytes(summary)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".tmp-{uuid.uuid4().hex[:8]}")
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise _collision(destination) from exc
        finally:
            temporary.unlink(missing_ok=True)
        fsync_directory(destination.parent)
    except StudyOutputCollisionError:
        raise
    except OSError as exc:
        raise StudyInfrastructureError(
            operation="write derived study summary",
            reason="ATOMIC_SUMMARY_PUBLICATION_FAILED",
            path=destination.name,
            expected="atomic publication to an absent explicit destination",
            actual=type(exc).__name__,
            action="Check the destination filesystem and retry with another absent output path.",
        ) from exc
    return len(payload)


def _json_bytes(summary: StudySummary) -> bytes:
    document = summary.as_dict()
    document.update({"derived": True, "canonical_authority": False})
    return (json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _csv_bytes(summary: StudySummary) -> bytes:
    rows = [row.as_dict() for row in summary.rows]
    row_fields = list(rows[0]) if rows else _empty_row_fields()
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=[*_TRACE_FIELDS, *row_fields], lineterminator="\n")
    writer.writeheader()
    trace = {
        "derived": True,
        "canonical_authority": False,
        "generated_at": summary.generated_at,
        "root_manifest_sha256": summary.root_manifest_sha256,
        "event_head_sequence": summary.event_head_sequence,
        "event_head_sha256": summary.event_head_sha256,
    }
    for row in rows:
        writer.writerow({**trace, **{key: _csv_value(value) for key, value in row.items()}})
    return output.getvalue().encode("utf-8")


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return value


def _empty_row_fields() -> list[str]:
    from dataclasses import fields

    from vamos.experiment.study.report_models import StudySummaryRow

    return [item.name for item in fields(StudySummaryRow)]


def _collision(path: Path) -> StudyOutputCollisionError:
    return StudyOutputCollisionError(
        operation="write derived study summary",
        reason="OUTPUT_COLLISION",
        path=path.name,
        expected="destination path that does not exist",
        actual="existing path",
        action="Choose another output path; summaries are never overwritten.",
    )


__all__ = ["SummaryFormat", "write_summary"]
