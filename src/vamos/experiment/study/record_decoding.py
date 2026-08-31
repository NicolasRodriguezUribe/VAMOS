"""Closed-schema task, attempt, and event record decoding."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from vamos.experiment.artifacts.models import deep_freeze

from .decoding import (
    _array,
    _decode_attempt_reference,
    _enum,
    _malformed,
    _object,
    _optional_string,
    _optional_uuid,
    _string,
    _verify_integrity,
)
from .models import AttemptRecord, Retryability, StudyEvent, TaskRecord
from .serialization import require_digest, require_fields, require_int, require_timestamp, require_uuid4, validate_header

_TASK_STATES = {"pending", "running", "succeeded", "failed", "interrupted", "cancelled", "skipped"}
_ATTEMPT_STATES = {"created", "running", "succeeded", "failed", "interrupted", "cancelled"}


def decode_task(value: Mapping[str, Any]) -> TaskRecord:
    role = "study_task"
    require_fields(
        value,
        {
            "document_type",
            "schema_version",
            "study_id",
            "task_id",
            "plan_index",
            "state",
            "attempts",
            "current_attempt_id",
            "selected_success_attempt_id",
            "retryability",
            "reason",
            "claim_epoch",
            "integrity",
        },
        role=role,
    )
    validate_header(value, role=role, document_type="vamos.study-task")
    attempts = tuple(_decode_attempt_reference(item, index) for index, item in enumerate(_array(value.get("attempts"), "$.attempts", role)))
    retry = _object(value.get("retryability"), "$.retryability", role)
    require_fields(retry, {"retryable", "category", "attempts_remaining"}, role=role)
    retryable = retry.get("retryable")
    if not isinstance(retryable, bool):
        _malformed(role, "INVALID_FIELD", "$.retryability.retryable", "boolean", retryable)
    reason = value.get("reason")
    if reason is not None:
        reason = deep_freeze(_object(reason, "$.reason", role))
    return TaskRecord(
        study_id=require_uuid4(value.get("study_id"), field="$.study_id", role=role),
        task_id=require_digest(value.get("task_id"), field="$.task_id", role=role, prefixed=True),
        plan_index=require_int(value.get("plan_index"), field="$.plan_index", role=role),
        state=cast(Any, _enum(value.get("state"), _TASK_STATES, "$.state", role)),
        attempts=attempts,
        current_attempt_id=_optional_uuid(value.get("current_attempt_id"), "$.current_attempt_id", role),
        selected_success_attempt_id=_optional_uuid(value.get("selected_success_attempt_id"), "$.selected_success_attempt_id", role),
        retryability=Retryability(
            retryable=retryable,
            category=_optional_string(retry.get("category"), "$.retryability.category", role),
            attempts_remaining=require_int(retry.get("attempts_remaining"), field="$.retryability.attempts_remaining", role=role),
        ),
        reason=cast(Any, reason),
        claim_epoch=require_int(value.get("claim_epoch"), field="$.claim_epoch", role=role),
        document_sha256=_verify_integrity(value, role),
    )


def decode_event(value: Mapping[str, Any], *, file_sha256: str = "") -> StudyEvent:
    role = "study_event"
    require_fields(
        value,
        {
            "document_type",
            "schema_version",
            "sequence",
            "event_id",
            "event_type",
            "entity",
            "transition",
            "execution_id",
            "timestamp",
            "reason",
            "payload",
            "previous_event_sha256",
            "integrity",
        },
        role=role,
    )
    validate_header(value, role=role, document_type="vamos.study-event")
    entity = _object(value.get("entity"), "$.entity", role)
    transition = _object(value.get("transition"), "$.transition", role)
    require_fields(entity, {"kind", "id"}, role=role)
    require_fields(transition, {"from", "to"}, role=role)
    previous = value.get("previous_event_sha256")
    if previous is not None:
        previous = require_digest(previous, field="$.previous_event_sha256", role=role)
    execution = value.get("execution_id")
    if execution is not None:
        execution = require_uuid4(execution, field="$.execution_id", role=role)
    reason = value.get("reason")
    if reason is not None:
        reason = deep_freeze(_object(reason, "$.reason", role))
    transition_from = transition.get("from")
    if transition_from is not None:
        transition_from = _string(transition_from, "$.transition.from", role)
    return StudyEvent(
        sequence=require_int(value.get("sequence"), field="$.sequence", role=role, minimum=1),
        event_id=require_uuid4(value.get("event_id"), field="$.event_id", role=role),
        event_type=_string(value.get("event_type"), "$.event_type", role),
        entity_kind=_string(entity.get("kind"), "$.entity.kind", role),
        entity_id=_string(entity.get("id"), "$.entity.id", role),
        transition_from=transition_from,
        transition_to=_string(transition.get("to"), "$.transition.to", role),
        execution_id=execution,
        timestamp=require_timestamp(value.get("timestamp"), field="$.timestamp", role=role),
        reason=cast(Any, reason),
        payload=deep_freeze(_object(value.get("payload"), "$.payload", role)),
        previous_event_sha256=previous,
        document_sha256=_verify_integrity(value, role),
        file_sha256=file_sha256,
    )


def decode_attempt(value: Mapping[str, Any]) -> AttemptRecord:
    role = "study_attempt"
    require_fields(
        value,
        {
            "document_type",
            "schema_version",
            "study_id",
            "task_id",
            "attempt_id",
            "attempt_number",
            "execution_id",
            "status",
            "timestamps",
            "lease_evidence",
            "failure",
            "run_reference",
            "integrity",
        },
        role=role,
    )
    validate_header(value, role=role, document_type="vamos.study-attempt")
    status = _enum(value.get("status"), _ATTEMPT_STATES, "$.status", role)
    timestamps = _decode_timestamps(value.get("timestamps"), status, role)
    lease = _decode_lease(value.get("lease_evidence"), role)
    failure = _decode_failure(value.get("failure"), status, role)
    run_reference = _decode_run_reference(value.get("run_reference"), role)
    return AttemptRecord(
        study_id=require_uuid4(value.get("study_id"), field="$.study_id", role=role),
        task_id=require_digest(value.get("task_id"), field="$.task_id", role=role, prefixed=True),
        attempt_id=require_uuid4(value.get("attempt_id"), field="$.attempt_id", role=role),
        attempt_number=require_int(value.get("attempt_number"), field="$.attempt_number", role=role, minimum=1),
        execution_id=require_uuid4(value.get("execution_id"), field="$.execution_id", role=role),
        status=cast(Any, status),
        timestamps=timestamps,
        lease_evidence=lease,
        failure=failure,
        run_reference=run_reference,
        document_sha256=_verify_integrity(value, role),
    )


def _decode_timestamps(value: object, status: str, role: str) -> Mapping[str, Any]:
    item = _object(value, "$.timestamps", role)
    require_fields(item, {"created_at", "started_at", "completed_at"}, role=role)
    require_timestamp(item.get("created_at"), field="$.timestamps.created_at", role=role)
    started = item.get("started_at")
    completed = item.get("completed_at")
    if started is not None:
        require_timestamp(started, field="$.timestamps.started_at", role=role)
    if completed is not None:
        require_timestamp(completed, field="$.timestamps.completed_at", role=role)
    terminal = status in {"succeeded", "failed", "interrupted", "cancelled"}
    if (status == "created" and (started is not None or completed is not None)) or (
        status == "running" and (started is None or completed is not None)
    ):
        _malformed(role, "INVALID_FIELD", "$.timestamps", f"timestamps consistent with {status}", item)
    if terminal and completed is None:
        _malformed(role, "INVALID_FIELD", "$.timestamps.completed_at", "terminal completion timestamp", completed)
    return cast(Mapping[str, Any], deep_freeze(item))


def _decode_lease(value: object, role: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    item = _object(value, "$.lease_evidence", role)
    require_fields(
        item,
        {"worker_id", "attempt_id", "claim_epoch", "token", "acquired_at", "heartbeat_at", "expires_at"},
        role=role,
    )
    require_uuid4(item.get("worker_id"), field="$.lease_evidence.worker_id", role=role)
    require_uuid4(item.get("attempt_id"), field="$.lease_evidence.attempt_id", role=role)
    require_int(item.get("claim_epoch"), field="$.lease_evidence.claim_epoch", role=role, minimum=1)
    _string(item.get("token"), "$.lease_evidence.token", role)
    for name in ("acquired_at", "heartbeat_at", "expires_at"):
        require_timestamp(item.get(name), field=f"$.lease_evidence.{name}", role=role)
    return cast(Mapping[str, Any], deep_freeze(item))


def _decode_failure(value: object, status: str, role: str) -> Mapping[str, Any] | None:
    if value is None:
        if status == "failed":
            _malformed(role, "INVALID_FIELD", "$.failure", "structured failure for failed attempt", None)
        return None
    item = _object(value, "$.failure", role)
    require_fields(item, {"category", "code", "message", "retryable", "safe_action"}, role=role)
    for name in ("category", "code", "message", "safe_action"):
        _string(item.get(name), f"$.failure.{name}", role)
    if not isinstance(item.get("retryable"), bool):
        _malformed(role, "INVALID_FIELD", "$.failure.retryable", "boolean", item.get("retryable"))
    if status != "failed":
        _malformed(role, "INVALID_FIELD", "$.failure", "null unless attempt failed", status)
    return cast(Mapping[str, Any], deep_freeze(item))


def _decode_run_reference(value: object, role: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    item = _object(value, "$.run_reference", role)
    require_fields(
        item,
        {"path", "role", "required_for", "semantic_sha256", "sha256", "bytes", "run_id", "task_id"},
        role=role,
    )
    _string(item.get("path"), "$.run_reference.path", role)
    if item.get("role") != "run_manifest":
        _malformed(role, "INVALID_FIELD", "$.run_reference.role", "run_manifest", item.get("role"))
    _array(item.get("required_for"), "$.run_reference.required_for", role)
    require_digest(item.get("semantic_sha256"), field="$.run_reference.semantic_sha256", role=role)
    require_digest(item.get("sha256"), field="$.run_reference.sha256", role=role)
    require_int(item.get("bytes"), field="$.run_reference.bytes", role=role)
    require_uuid4(item.get("run_id"), field="$.run_reference.run_id", role=role)
    require_digest(item.get("task_id"), field="$.run_reference.task_id", role=role, prefixed=True)
    return cast(Mapping[str, Any], deep_freeze(item))


__all__ = ["decode_attempt", "decode_event", "decode_task"]
