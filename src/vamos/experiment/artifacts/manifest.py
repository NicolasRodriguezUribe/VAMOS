"""RunManifest parsing, semantic validation, and self-integrity checks."""

from __future__ import annotations

import hmac
import re
import uuid
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn

from .errors import (
    ArtifactIntegrityError,
    ArtifactResourceLimitError,
    ManifestValidationError,
    MissingManifestFieldError,
    UnsupportedSchemaError,
)
from .jsonio import canonical_json_bytes, load_json_file, manifest_self_hash, sha256_bytes
from .lineage import validate_replay_lineage
from .models import ArtifactDescriptor, LoadLimits, ResolvedRunSpec, RunManifest, deep_freeze
from .paths import validate_relative_artifact_path

DOCUMENT_TYPE = "vamos.run-manifest"
SCHEMA_VERSION = "1.0.0"
RESOLVED_SPEC_VERSION = "vamos.resolved-run-spec/1"
TERMINAL_STATUSES = ("succeeded", "failed", "partial", "cancelled")
ALL_STATUSES = ("running", *TERMINAL_STATUSES)
KNOWN_REQUIRED_FOR = ("load", "inspect", "verify", "replay", "analysis")
SINGLETON_ARTIFACT_ROLES = ("environment", "result_bundle", "metrics", "events")


def parse_run_manifest(path: Path, *, limits: LoadLimits, operation: str) -> RunManifest:
    value = load_json_file(
        path,
        operation=operation,
        artifact_role="manifest",
        max_bytes=limits.max_manifest_bytes,
        max_depth=limits.max_json_depth,
    )
    return validate_run_manifest(value, limits=limits, operation=operation, path=path)


def validate_run_manifest(
    value: Mapping[str, Any],
    *,
    limits: LoadLimits,
    operation: str,
    path: Path | str = "manifest.json",
) -> RunManifest:
    document_type = _required(value, "document_type", operation=operation, path=path)
    if document_type != DOCUMENT_TYPE:
        raise UnsupportedSchemaError(
            operation=operation,
            field="$.document_type",
            path=path,
            reason="does not identify a VAMOS v1 run manifest",
            expected=DOCUMENT_TYPE,
            actual=document_type,
            action="This is a pre-release format; regenerate the run with the current VAMOS version.",
        )
    schema_version = _required(value, "schema_version", operation=operation, path=path)
    if schema_version != SCHEMA_VERSION:
        raise UnsupportedSchemaError(
            operation=operation,
            field="$.schema_version",
            path=path,
            reason="is not supported by this reader",
            expected=SCHEMA_VERSION,
            actual=schema_version,
            action="This is a pre-release format; regenerate the run with the current VAMOS version.",
        )
    run_id = _required_string(value, "run_id", operation=operation, path=path)
    _validate_uuid4(run_id, operation=operation, path=path, field="$.run_id")
    retry_id = value.get("retry_of_run_id")
    if retry_id is not None:
        if not isinstance(retry_id, str):
            _invalid(operation, path, "$.retry_of_run_id", "UUID string or null", retry_id)
        _validate_uuid4(retry_id, operation=operation, path=path, field="$.retry_of_run_id")
        if retry_id == run_id:
            _invalid(operation, path, "$.retry_of_run_id", "UUID different from run_id", retry_id)
    status = _required_string(value, "status", operation=operation, path=path)
    if status not in ALL_STATUSES:
        _invalid(operation, path, "$.status", ALL_STATUSES, status)
    timestamps = _required_mapping(value, "timestamps", operation=operation, path=path)
    _validate_timestamp(_required_string(timestamps, "started_at", operation=operation, path=path, prefix="$.timestamps"), operation, path)
    completed_at = timestamps.get("completed_at")
    if status in TERMINAL_STATUSES:
        if not isinstance(completed_at, str):
            _invalid(operation, path, "$.timestamps.completed_at", "RFC 3339 timestamp for a terminal run", completed_at)
        assert isinstance(completed_at, str)
        _validate_timestamp(completed_at, operation, path)
    elif completed_at is not None:
        _invalid(operation, path, "$.timestamps.completed_at", "absent for status=running", completed_at)
    _required_mapping(value, "requested_spec", operation=operation, path=path)
    resolved_raw = _required_mapping(value, "resolved_spec", operation=operation, path=path)
    _validate_resolved_spec(resolved_raw, operation=operation, path=path)
    expected_task = "sha256:" + sha256_bytes(canonical_json_bytes(resolved_raw))
    task_id = _required_string(value, "task_id", operation=operation, path=path)
    if not hmac.compare_digest(task_id, expected_task):
        _invalid(operation, path, "$.task_id", expected_task, task_id, reason="does not match canonical resolved_spec")
    provenance = _required_mapping(value, "provenance", operation=operation, path=path)
    provenance_timestamps = provenance.get("timestamps")
    if provenance_timestamps != timestamps:
        _invalid(
            operation,
            path,
            "$.provenance.timestamps",
            timestamps,
            provenance_timestamps,
            reason="does not match top-level timestamps",
        )
    replayability = _required_mapping(value, "replayability", operation=operation, path=path)
    if replayability.get("declared_level") not in ("exact", "compatible", "best_effort", "manual", "unavailable"):
        _invalid(
            operation,
            path,
            "$.replayability.declared_level",
            ("exact", "compatible", "best_effort", "manual", "unavailable"),
            replayability.get("declared_level"),
        )
    artifacts_raw = value.get("artifacts")
    if not isinstance(artifacts_raw, list):
        _invalid(operation, path, "$.artifacts", "JSON array", type(artifacts_raw).__name__)
    if len(artifacts_raw) > limits.max_artifacts:
        raise ArtifactResourceLimitError(
            operation=operation,
            limit="max_artifacts",
            configured=limits.max_artifacts,
            observed=len(artifacts_raw),
            artifact_role="manifest",
            path=path,
            action="Inspect the source and pass explicit trusted LoadLimits only if this inventory is expected.",
        )
    descriptors = _validate_descriptors(artifacts_raw, operation=operation, path=path)
    roles = {descriptor.role for descriptor in descriptors}
    if status == "succeeded" and "result_bundle" not in roles:
        _invalid(operation, path, "$.artifacts", "succeeded run containing result_bundle", sorted(roles))
    if status == "succeeded" and "environment" not in roles:
        _invalid(operation, path, "$.artifacts", "succeeded run containing environment", sorted(roles))
    if status in TERMINAL_STATUSES:
        _required_mapping(value, "outcome", operation=operation, path=path)
        integrity = _required_mapping(value, "integrity", operation=operation, path=path)
        expected_self_hash = integrity.get("manifest_sha256")
        if not isinstance(expected_self_hash, str) or re.fullmatch(r"[0-9a-f]{64}", expected_self_hash) is None:
            _invalid(operation, path, "$.integrity.manifest_sha256", "64 lowercase hexadecimal characters", expected_self_hash)
        actual_self_hash = manifest_self_hash(value)
        if not hmac.compare_digest(expected_self_hash, actual_self_hash):
            raise ArtifactIntegrityError(
                operation=operation,
                artifact_role="manifest",
                path=path,
                field="$.integrity.manifest_sha256",
                reason="does not match the canonical manifest content",
                expected=expected_self_hash,
                actual=actual_self_hash,
                expected_sha256=expected_self_hash,
                actual_sha256=actual_self_hash,
                state="hash_mismatch",
                action="Restore manifest.json from the original run; formatting-only edits may retain the existing semantic hash.",
            )
    if status == "failed" and not isinstance(value.get("failure"), Mapping):
        _invalid(operation, path, "$.failure", "structured failure object for status=failed", value.get("failure"))
    if "labels" in value:
        labels = value["labels"]
        if not isinstance(labels, Mapping) or any(not isinstance(key, str) or not isinstance(item, str) for key, item in labels.items()):
            _invalid(operation, path, "$.labels", "object of string keys and values", labels)
    validate_replay_lineage(value.get("lineage"), run_id=run_id, status=status, operation=operation, path=path)
    frozen = deep_freeze(value)
    if not isinstance(frozen, Mapping):
        raise AssertionError("deep_freeze returned a non-mapping for a manifest")
    return RunManifest(
        frozen,
        ResolvedRunSpec.from_mapping(resolved_raw),
        tuple(descriptors),
    )


def build_terminal_manifest(value: Mapping[str, Any], *, limits: LoadLimits) -> RunManifest:
    payload = dict(value)
    payload["integrity"] = {}
    payload["integrity"] = {"manifest_sha256": manifest_self_hash(payload)}
    return validate_run_manifest(payload, limits=limits, operation="save result", path="manifest.json")


def _validate_resolved_spec(value: Mapping[str, Any], *, operation: str, path: Path | str) -> None:
    if value.get("spec_version") != RESOLVED_SPEC_VERSION:
        _invalid(operation, path, "$.resolved_spec.spec_version", RESOLVED_SPEC_VERSION, value.get("spec_version"))
    for field in (
        "problem",
        "algorithm",
        "operators",
        "backend",
        "termination",
        "seed",
        "population",
        "defaults_applied",
        "determinism",
    ):
        if field not in value:
            _invalid(operation, path, f"$.resolved_spec.{field}", "required field", "missing")
    if isinstance(value.get("seed"), bool) or not isinstance(value.get("seed"), int):
        _invalid(operation, path, "$.resolved_spec.seed", "integer seed", value.get("seed"))
    for descriptor_name in ("problem", "algorithm", "termination"):
        descriptor = value.get(descriptor_name)
        if not isinstance(descriptor, Mapping):
            _invalid(operation, path, f"$.resolved_spec.{descriptor_name}", "component descriptor object", descriptor)
        for key in ("kind", "component_id", "provider", "config", "resolution"):
            if key not in descriptor:
                _invalid(operation, path, f"$.resolved_spec.{descriptor_name}.{key}", "required field", "missing")
    backend = value.get("backend")
    if (
        not isinstance(backend, Mapping)
        or not isinstance(backend.get("kernel"), Mapping)
        or not isinstance(backend.get("evaluation"), Mapping)
    ):
        _invalid(operation, path, "$.resolved_spec.backend", "kernel and evaluation descriptors", backend)
    if not isinstance(value.get("operators"), Mapping):
        _invalid(operation, path, "$.resolved_spec.operators", "JSON object", value.get("operators"))
    if not isinstance(value.get("defaults_applied"), list):
        _invalid(operation, path, "$.resolved_spec.defaults_applied", "JSON array", value.get("defaults_applied"))


def _validate_descriptors(
    values: list[Any],
    *,
    operation: str,
    path: Path | str,
) -> list[ArtifactDescriptor]:
    descriptors: list[ArtifactDescriptor] = []
    singleton_roles: set[str] = set()
    referenced_paths: set[str] = set()
    for index, item in enumerate(values):
        prefix = f"$.artifacts[{index}]"
        if not isinstance(item, Mapping):
            _invalid(operation, path, prefix, "artifact descriptor object", type(item).__name__)
        role = _required_string(item, "role", operation=operation, path=path, prefix=prefix)
        artifact_path = validate_relative_artifact_path(
            _required_string(item, "path", operation=operation, path=path, prefix=prefix),
            role=role,
            operation=operation,
        )
        if artifact_path in referenced_paths:
            _invalid(operation, path, f"{prefix}.path", "path referenced once", artifact_path, reason="duplicates another artifact path")
        referenced_paths.add(artifact_path)
        if role in SINGLETON_ARTIFACT_ROLES:
            if role in singleton_roles:
                _invalid(operation, path, f"{prefix}.role", "singleton role referenced once", role)
            singleton_roles.add(role)
        media_type = _required_string(item, "media_type", operation=operation, path=path, prefix=prefix)
        digest = _required_string(item, "sha256", operation=operation, path=path, prefix=prefix)
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            _invalid(operation, path, f"{prefix}.sha256", "64 lowercase hexadecimal characters", digest)
        byte_length = item.get("bytes")
        if isinstance(byte_length, bool) or not isinstance(byte_length, int) or byte_length < 0:
            _invalid(operation, path, f"{prefix}.bytes", "non-negative integer", byte_length)
        required_for = item.get("required_for")
        if not isinstance(required_for, list) or any(value not in KNOWN_REQUIRED_FOR for value in required_for):
            _invalid(operation, path, f"{prefix}.required_for", f"array containing only {KNOWN_REQUIRED_FOR}", required_for)
        if len(required_for) != len(set(required_for)):
            _invalid(operation, path, f"{prefix}.required_for", "array without duplicate purposes", required_for)
        canonical = item.get("canonical")
        if not isinstance(canonical, bool):
            _invalid(operation, path, f"{prefix}.canonical", "boolean", canonical)
        array_contract = item.get("array_contract")
        if array_contract is not None and not isinstance(array_contract, Mapping):
            _invalid(operation, path, f"{prefix}.array_contract", "object or absent", array_contract)
        descriptors.append(
            ArtifactDescriptor(
                role=role,
                path=artifact_path,
                media_type=media_type,
                sha256=digest,
                bytes=byte_length,
                required_for=tuple(required_for),
                canonical=canonical,
                array_contract=deep_freeze(array_contract) if array_contract is not None else None,
            )
        )
    return descriptors


def _required(
    value: Mapping[str, Any],
    key: str,
    *,
    operation: str,
    path: Path | str,
    prefix: str = "$",
) -> Any:
    if key not in value:
        raise MissingManifestFieldError(
            operation=operation,
            path=path,
            field=f"{prefix}.{key}",
            reason="is missing",
            expected="required v1 manifest field",
            actual="missing",
            action="Restore or regenerate the complete v1 manifest.",
        )
    return value[key]


def _required_string(
    value: Mapping[str, Any],
    key: str,
    *,
    operation: str,
    path: Path | str,
    prefix: str = "$",
) -> str:
    item = _required(value, key, operation=operation, path=path, prefix=prefix)
    if not isinstance(item, str) or not item:
        _invalid(operation, path, f"{prefix}.{key}", "non-empty string", item)
    return item


def _required_mapping(
    value: Mapping[str, Any],
    key: str,
    *,
    operation: str,
    path: Path | str,
    prefix: str = "$",
) -> Mapping[str, Any]:
    item = _required(value, key, operation=operation, path=path, prefix=prefix)
    if not isinstance(item, Mapping):
        _invalid(operation, path, f"{prefix}.{key}", "JSON object", type(item).__name__)
    return item


def _validate_uuid4(value: str, *, operation: str, path: Path | str, field: str) -> None:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ManifestValidationError(
            operation=operation,
            field=field,
            path=path,
            reason="is not a valid UUID",
            expected="lowercase UUIDv4 string",
            actual=value,
            action="Restore the original manifest identity.",
        ) from exc
    if parsed.version != 4 or str(parsed) != value:
        _invalid(operation, path, field, "lowercase UUIDv4 string", value)


def _validate_timestamp(value: str, operation: str, path: Path | str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ManifestValidationError(
            operation=operation,
            field="$.timestamps",
            path=path,
            reason="contains an invalid timestamp",
            expected="RFC 3339 timestamp with an explicit offset",
            actual=value,
            action="Restore the timestamp from the original manifest.",
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _invalid(operation, path, "$.timestamps", "RFC 3339 timestamp with explicit offset", value)


def _invalid(
    operation: str,
    path: Path | str,
    field: str,
    expected: Any,
    actual: Any,
    *,
    reason: str = "is invalid",
) -> NoReturn:
    raise ManifestValidationError(
        operation=operation,
        field=field,
        path=path,
        reason=reason,
        expected=expected,
        actual=actual,
        action="Restore or regenerate the artifact with a supported VAMOS v1 writer.",
    )


__all__ = [
    "DOCUMENT_TYPE",
    "RESOLVED_SPEC_VERSION",
    "SCHEMA_VERSION",
    "build_terminal_manifest",
    "parse_run_manifest",
    "validate_run_manifest",
]
