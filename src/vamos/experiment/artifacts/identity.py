"""Content-derived identities shared by run and study artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .jsonio import canonical_json_bytes, normalize_json, sha256_bytes


def resolved_spec_task_digest(resolved_spec: Mapping[str, Any]) -> str:
    """Return the canonical RunManifest task digest for a resolved run spec."""
    normalized = normalize_json(resolved_spec, field="$.resolved_spec")
    if not isinstance(normalized, dict):
        raise AssertionError("a resolved run specification must be an object")
    return sha256_bytes(canonical_json_bytes(normalized))


def resolved_spec_task_id(resolved_spec: Mapping[str, Any]) -> str:
    """Return ``sha256:<digest>`` for the complete resolved run specification."""
    return f"sha256:{resolved_spec_task_digest(resolved_spec)}"


__all__ = ["resolved_spec_task_digest", "resolved_spec_task_id"]
