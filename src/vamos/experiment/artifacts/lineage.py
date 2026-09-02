"""Focused schema-1 replay-lineage validation."""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import NoReturn

from .errors import ManifestValidationError

MAX_REPLAY_LINEAGE_DEPTH = 64


def validate_replay_lineage(
    value: object,
    *,
    run_id: str,
    status: str,
    operation: str,
    path: Path | str,
) -> None:
    """Validate optional replay lineage without expanding the central validator."""
    if value is None:
        return
    if not isinstance(value, Mapping):
        _invalid(operation, path, "$.lineage", "object or absent", value)
    if value.get("execution_kind") != "replay":
        _invalid(operation, path, "$.lineage.execution_kind", "replay", value.get("execution_kind"))
    source_run_id = _uuid(value.get("source_run_id"), operation, path, "$.lineage.source_run_id")
    root_run_id = _uuid(value.get("root_run_id"), operation, path, "$.lineage.root_run_id")
    if source_run_id == run_id or root_run_id == run_id:
        _invalid(operation, path, "$.lineage", "ancestor IDs different from run_id", value)
    _digest(value.get("source_manifest_sha256"), operation, path, "$.lineage.source_manifest_sha256")
    _digest(value.get("replay_plan_sha256"), operation, path, "$.lineage.replay_plan_sha256")
    depth = value.get("depth")
    if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= MAX_REPLAY_LINEAGE_DEPTH:
        _invalid(operation, path, "$.lineage.depth", f"integer from 1 through {MAX_REPLAY_LINEAGE_DEPTH}", depth)
    if value.get("compatibility_level") != "exact":
        _invalid(operation, path, "$.lineage.compatibility_level", "exact", value.get("compatibility_level"))
    comparison = value.get("comparison")
    if not isinstance(comparison, Mapping):
        _invalid(operation, path, "$.lineage.comparison", "comparison object", comparison)
    comparison_status = comparison.get("status")
    allowed = ("execution_failed",) if status == "failed" else ("exact_match", "mismatch")
    if comparison_status not in allowed:
        _invalid(operation, path, "$.lineage.comparison.status", allowed, comparison_status)
    arrays = comparison.get("arrays")
    if not isinstance(arrays, list):
        _invalid(operation, path, "$.lineage.comparison.arrays", "array", arrays)


def _uuid(value: object, operation: str, path: Path | str, field: str) -> str:
    if not isinstance(value, str):
        _invalid(operation, path, field, "lowercase UUIDv4", value)
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ManifestValidationError(
            operation=operation,
            path=path,
            field=field,
            reason="is not a valid UUID",
            expected="lowercase UUIDv4",
            actual=value,
            action="Restore the canonical replay lineage.",
        ) from exc
    if parsed.version != 4 or str(parsed) != value:
        _invalid(operation, path, field, "lowercase UUIDv4", value)
    return value


def _digest(value: object, operation: str, path: Path | str, field: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        _invalid(operation, path, field, "64 lowercase hexadecimal characters", value)


def _invalid(operation: str, path: Path | str, field: str, expected: object, actual: object) -> NoReturn:
    raise ManifestValidationError(
        operation=operation,
        path=path,
        field=field,
        reason="is invalid",
        expected=expected,
        actual=actual,
        action="Restore or regenerate the canonical schema-1 replay run.",
    )


__all__ = ["MAX_REPLAY_LINEAGE_DEPTH", "validate_replay_lineage"]
