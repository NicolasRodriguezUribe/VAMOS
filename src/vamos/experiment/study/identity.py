"""Scientific and entity identity rules for StudyManifest v1."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping
from typing import Any

from vamos.experiment.artifacts.identity import resolved_spec_task_digest, resolved_spec_task_id
from vamos.experiment.artifacts.jsonio import sha256_bytes

from .models import SCHEMA_VERSION
from .serialization import canonical_json


def new_uuid4() -> str:
    """Return the canonical lowercase UUIDv4 representation."""
    return str(uuid.uuid4())


def compute_task_digest(resolved_spec: Mapping[str, Any]) -> str:
    """Delegate to the sole RunManifest task-identity implementation."""
    return resolved_spec_task_digest(resolved_spec)


def compute_task_id(resolved_spec: Mapping[str, Any]) -> str:
    """Delegate to the sole RunManifest task-identity implementation."""
    return resolved_spec_task_id(resolved_spec)


def compute_plan_id(task_ids: Iterable[str]) -> str:
    """Hash the schema-tagged, canonically sorted scientific task set."""
    projections = [{"task_id": task_id} for task_id in sorted(task_ids)]
    payload = {
        "document_type": "vamos.resolved-study-plan",
        "schema_version": SCHEMA_VERSION,
        "tasks": projections,
    }
    return f"sha256:{sha256_bytes(canonical_json(payload))}"


__all__ = ["compute_plan_id", "compute_task_digest", "compute_task_id", "new_uuid4"]
