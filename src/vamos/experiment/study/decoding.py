"""Closed-schema decoding for canonical StudyManifest v1 documents."""

from __future__ import annotations

import hmac
from collections.abc import Mapping
from typing import Any, NoReturn, cast

from vamos.experiment.artifacts.errors import RunArtifactError
from vamos.experiment.artifacts.manifest import validate_resolved_run_spec
from vamos.experiment.artifacts.models import deep_freeze

from .errors import InvalidStudySpecError, MalformedStudyError, PlanMismatchError, StudyIntegrityError
from .identity import compute_plan_id, compute_task_digest, compute_task_id
from .models import (
    AttemptReference,
    DocumentReference,
    PlanTask,
    ResolvedStudyPlan,
    StudyCounts,
    StudyManifest,
    StudySpec,
)
from .serialization import (
    document_self_hash,
    require_digest,
    require_fields,
    require_int,
    require_timestamp,
    require_uuid4,
    validate_header,
)

_INTEGRITY_FIELDS = {"document_sha256"}
_REFERENCE_FIELDS = {"path", "role", "required_for", "semantic_sha256", "sha256", "bytes"}
_ATTEMPT_REFERENCE_FIELDS = _REFERENCE_FIELDS | {"attempt_id", "attempt_number"}
_STUDY_STATES = {"created", "running", "paused", "completed", "completed_with_failures", "failed", "cancelled"}


def decode_spec(value: Mapping[str, Any]) -> StudySpec:
    role = "study_spec"
    require_fields(
        value,
        {"document_type", "schema_version", "study_id", "matrix", "run_defaults", "policy", "labels", "metadata", "integrity"},
        role=role,
    )
    validate_header(value, role=role, document_type="vamos.study-spec")
    study_id = require_uuid4(value.get("study_id"), field="$.study_id", role=role)
    matrix = _object(value.get("matrix"), "$.matrix", role)
    require_fields(matrix, {"problems", "algorithms", "seeds"}, role=role)
    defaults = _object(value.get("run_defaults"), "$.run_defaults", role)
    require_fields(
        defaults,
        {"max_evaluations", "pop_size", "engine", "eval_strategy", "n_var", "n_obj", "problem_kwargs", "algorithm_configs"},
        role=role,
    )
    policy = _object(value.get("policy"), "$.policy", role)
    require_fields(policy, {"on_error", "max_attempts_per_task"}, role=role)
    try:
        spec = StudySpec(
            problems=_string_array(matrix.get("problems"), "$.matrix.problems", role),
            algorithms=_string_array(matrix.get("algorithms"), "$.matrix.algorithms", role),
            seeds=_int_array(matrix.get("seeds"), "$.matrix.seeds", role),
            max_evaluations=_optional_int(defaults.get("max_evaluations"), "$.run_defaults.max_evaluations", role),
            pop_size=_optional_int(defaults.get("pop_size"), "$.run_defaults.pop_size", role),
            engine=_optional_string(defaults.get("engine"), "$.run_defaults.engine", role),
            eval_strategy=_string(defaults.get("eval_strategy"), "$.run_defaults.eval_strategy", role),
            n_var=_optional_int(defaults.get("n_var"), "$.run_defaults.n_var", role),
            n_obj=_optional_int(defaults.get("n_obj"), "$.run_defaults.n_obj", role),
            problem_kwargs=_object(defaults.get("problem_kwargs"), "$.run_defaults.problem_kwargs", role),
            algorithm_configs=_object(defaults.get("algorithm_configs"), "$.run_defaults.algorithm_configs", role),
            on_error=cast(Any, policy.get("on_error")),
            max_attempts_per_task=require_int(
                policy.get("max_attempts_per_task"), field="$.policy.max_attempts_per_task", role=role, minimum=1
            ),
            labels=_object(value.get("labels"), "$.labels", role),
            metadata=_object(value.get("metadata"), "$.metadata", role),
        )
    except InvalidStudySpecError as exc:
        raise MalformedStudyError(
            operation="load study",
            reason=exc.reason,
            document_role=role,
            field=exc.field,
            expected=exc.expected,
            actual=exc.actual,
            action="Restore a valid canonical study specification.",
        ) from exc
    object.__setattr__(spec, "study_id", study_id)
    _verify_integrity(value, role)
    return spec


def decode_plan(value: Mapping[str, Any]) -> ResolvedStudyPlan:
    role = "resolved_plan"
    require_fields(value, {"document_type", "schema_version", "plan_id", "task_count", "tasks", "integrity"}, role=role)
    validate_header(value, role=role, document_type="vamos.resolved-study-plan")
    plan_id = require_digest(value.get("plan_id"), field="$.plan_id", role=role, prefixed=True)
    count = require_int(value.get("task_count"), field="$.task_count", role=role)
    raw_tasks = _array(value.get("tasks"), "$.tasks", role)
    if len(raw_tasks) != count:
        _malformed(role, "COUNT_MISMATCH", "$.task_count", len(raw_tasks), count)
    tasks = tuple(_decode_plan_task(item, index) for index, item in enumerate(raw_tasks))
    if tuple(task.task_id for task in tasks) != tuple(sorted(task.task_id for task in tasks)):
        _malformed(role, "PLAN_MISMATCH", "$.tasks", "ascending task_id order", [task.task_id for task in tasks])
    if sorted(task.plan_index for task in tasks) != list(range(count)):
        _malformed(role, "PLAN_MISMATCH", "$.tasks[*].plan_index", list(range(count)), [task.plan_index for task in tasks])
    expected_plan = compute_plan_id(task.task_id for task in tasks)
    if not hmac.compare_digest(plan_id, expected_plan):
        raise PlanMismatchError(
            operation="load study",
            reason="PLAN_MISMATCH",
            document_role=role,
            field="$.plan_id",
            expected=expected_plan,
            actual=plan_id,
            action="Restore the immutable published plan or create a new study for changed science.",
        )
    integrity = _verify_integrity(value, role)
    return ResolvedStudyPlan(plan_id=plan_id, tasks=tasks, document_sha256=integrity)


def _decode_plan_task(value: object, index: int) -> PlanTask:
    role = "resolved_plan"
    item = _object(value, f"$.tasks[{index}]", role)
    require_fields(item, {"plan_index", "requested_run", "resolved_run_spec", "task_id", "task_spec_sha256"}, role=role)
    plan_index = require_int(item.get("plan_index"), field=f"$.tasks[{index}].plan_index", role=role)
    requested = _object(item.get("requested_run"), f"$.tasks[{index}].requested_run", role)
    resolved = _object(item.get("resolved_run_spec"), f"$.tasks[{index}].resolved_run_spec", role)
    try:
        validate_resolved_run_spec(resolved, operation="load study", path=f"plan task {index}")
    except RunArtifactError as exc:
        raise MalformedStudyError(
            operation="load study",
            reason="INVALID_RESOLVED_RUN_SPEC",
            document_role=role,
            field=f"$.tasks[{index}].resolved_run_spec",
            expected=exc.expected,
            actual=exc.actual,
            action="Restore the fully resolved V1 run specification.",
        ) from exc
    task_id = require_digest(item.get("task_id"), field=f"$.tasks[{index}].task_id", role=role, prefixed=True)
    digest = require_digest(item.get("task_spec_sha256"), field=f"$.tasks[{index}].task_spec_sha256", role=role)
    expected_id = compute_task_id(resolved)
    expected_digest = compute_task_digest(resolved)
    if task_id != expected_id or digest != expected_digest:
        raise PlanMismatchError(
            operation="load study",
            reason="TASK_ID_MISMATCH",
            document_role=role,
            field=f"$.tasks[{index}]",
            expected={"task_id": expected_id, "task_spec_sha256": expected_digest},
            actual={"task_id": task_id, "task_spec_sha256": digest},
            action="Restore the immutable plan; a changed resolved spec requires a new study.",
        )
    return PlanTask(plan_index, deep_freeze(requested), deep_freeze(resolved), task_id, digest)


def decode_manifest(value: Mapping[str, Any]) -> StudyManifest:
    role = "study_manifest"
    require_fields(
        value,
        {
            "document_type",
            "schema_version",
            "study_id",
            "plan_id",
            "state",
            "created_at",
            "updated_at",
            "execution_id",
            "policy",
            "spec",
            "plan",
            "counts",
            "checkpoint",
            "integrity",
        },
        role=role,
    )
    validate_header(value, role=role, document_type="vamos.study-manifest")
    study_id = require_uuid4(value.get("study_id"), field="$.study_id", role=role)
    plan_id = require_digest(value.get("plan_id"), field="$.plan_id", role=role, prefixed=True)
    state = _enum(value.get("state"), _STUDY_STATES, "$.state", role)
    policy = _object(value.get("policy"), "$.policy", role)
    require_fields(policy, {"on_error", "max_attempts_per_task"}, role=role)
    on_error = _enum(policy.get("on_error"), {"fail_fast", "continue"}, "$.policy.on_error", role)
    counts = _decode_counts(value.get("counts"), role)
    checkpoint = _object(value.get("checkpoint"), "$.checkpoint", role)
    require_fields(checkpoint, {"sequence", "event_sha256"}, role=role)
    integrity = _verify_integrity(value, role)
    execution_id = value.get("execution_id")
    if execution_id is not None:
        execution_id = require_uuid4(execution_id, field="$.execution_id", role=role)
    return StudyManifest(
        study_id=study_id,
        plan_id=plan_id,
        state=cast(Any, state),
        created_at=require_timestamp(value.get("created_at"), field="$.created_at", role=role),
        updated_at=require_timestamp(value.get("updated_at"), field="$.updated_at", role=role),
        execution_id=execution_id,
        on_error=cast(Any, on_error),
        max_attempts_per_task=require_int(
            policy.get("max_attempts_per_task"), field="$.policy.max_attempts_per_task", role=role, minimum=1
        ),
        spec=_decode_reference(value.get("spec"), role),
        plan=_decode_reference(value.get("plan"), role),
        counts=counts,
        checkpoint_sequence=require_int(checkpoint.get("sequence"), field="$.checkpoint.sequence", role=role, minimum=1),
        checkpoint_event_sha256=require_digest(checkpoint.get("event_sha256"), field="$.checkpoint.event_sha256", role=role),
        document_sha256=integrity,
    )


def _decode_reference(value: object, role: str) -> DocumentReference:
    item = _object(value, "$.reference", role)
    require_fields(item, _REFERENCE_FIELDS, role=role)
    return DocumentReference(
        path=_string(item.get("path"), "$.reference.path", role),
        role=_string(item.get("role"), "$.reference.role", role),
        required_for=tuple(_string_array(item.get("required_for"), "$.reference.required_for", role)),
        semantic_sha256=require_digest(item.get("semantic_sha256"), field="$.reference.semantic_sha256", role=role),
        sha256=require_digest(item.get("sha256"), field="$.reference.sha256", role=role),
        bytes=require_int(item.get("bytes"), field="$.reference.bytes", role=role),
    )


def _decode_attempt_reference(value: object, index: int) -> AttemptReference:
    role = "study_task"
    item = _object(value, f"$.attempts[{index}]", role)
    require_fields(item, _ATTEMPT_REFERENCE_FIELDS, role=role)
    base = _decode_reference({key: item[key] for key in _REFERENCE_FIELDS}, role)
    return AttemptReference(
        attempt_id=require_uuid4(item.get("attempt_id"), field=f"$.attempts[{index}].attempt_id", role=role),
        attempt_number=require_int(item.get("attempt_number"), field=f"$.attempts[{index}].attempt_number", role=role, minimum=1),
        path=base.path,
        role=base.role,
        required_for=base.required_for,
        semantic_sha256=base.semantic_sha256,
        sha256=base.sha256,
        bytes=base.bytes,
    )


def _decode_counts(value: object, role: str) -> StudyCounts:
    item = _object(value, "$.counts", role)
    fields = {"tasks", "pending", "running", "succeeded", "failed", "interrupted", "cancelled", "skipped"}
    require_fields(item, fields, role=role)
    parsed = {name: require_int(item.get(name), field=f"$.counts.{name}", role=role) for name in fields}
    return StudyCounts(**parsed)


def _verify_integrity(value: Mapping[str, Any], role: str) -> str:
    integrity = _object(value.get("integrity"), "$.integrity", role)
    require_fields(integrity, _INTEGRITY_FIELDS, role=role)
    actual = require_digest(integrity.get("document_sha256"), field="$.integrity.document_sha256", role=role)
    expected = document_self_hash(value)
    if not hmac.compare_digest(actual, expected):
        raise StudyIntegrityError(
            operation="load study",
            reason="DOCUMENT_HASH_MISMATCH",
            document_role=role,
            field="$.integrity.document_sha256",
            expected=expected,
            actual=actual,
            action="Restore the canonical document; VAMOS will not repair or reinterpret corruption.",
        )
    return actual


def _object(value: object, field: str, role: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _malformed(role, "INVALID_FIELD", field, "JSON object", type(value).__name__)
    return dict(cast(Mapping[str, Any], value))


def _optional_object(value: object, field: str, role: str) -> Mapping[str, Any] | None:
    return None if value is None else deep_freeze(_object(value, field, role))


def _array(value: object, field: str, role: str) -> list[Any]:
    if not isinstance(value, list):
        _malformed(role, "INVALID_FIELD", field, "JSON array", type(value).__name__)
    return value


def _string(value: object, field: str, role: str) -> str:
    if not isinstance(value, str) or not value:
        _malformed(role, "INVALID_FIELD", field, "non-empty string", value)
    return value


def _optional_string(value: object, field: str, role: str) -> str | None:
    return None if value is None else _string(value, field, role)


def _optional_uuid(value: object, field: str, role: str) -> str | None:
    return None if value is None else require_uuid4(value, field=field, role=role)


def _optional_int(value: object, field: str, role: str) -> int | None:
    return None if value is None else require_int(value, field=field, role=role, minimum=1)


def _string_array(value: object, field: str, role: str) -> list[str]:
    return [_string(item, f"{field}[{index}]", role) for index, item in enumerate(_array(value, field, role))]


def _int_array(value: object, field: str, role: str) -> list[int]:
    result: list[int] = []
    for index, item in enumerate(_array(value, field, role)):
        if isinstance(item, bool) or not isinstance(item, int):
            _malformed(role, "INVALID_FIELD", f"{field}[{index}]", "integer", item)
        result.append(item)
    return result


def _enum(value: object, allowed: set[str], field: str, role: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        _malformed(role, "INVALID_FIELD", field, sorted(allowed), value)
    return value


def _malformed(role: str, reason: str, field: str, expected: object, actual: object) -> NoReturn:
    raise MalformedStudyError(
        operation="load study",
        reason=reason,
        document_role=role,
        field=field,
        expected=expected,
        actual=actual,
        action="Restore the canonical V1 document from a trusted copy.",
    )


__all__ = ["decode_manifest", "decode_plan", "decode_spec"]
