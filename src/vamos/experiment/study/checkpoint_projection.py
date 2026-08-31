"""Checkpoint validation and immutable effective-view projection."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from vamos.experiment.artifacts.jsonio import sha256_bytes

from .documents import attempt_document
from .errors import StudyCheckpointError, StudyIntegrityError
from .models import (
    AttemptRecord,
    AttemptReference,
    AttemptState,
    StudyCounts,
    StudyEvent,
    StudyManifest,
    StudyState,
    TaskRecord,
    TaskState,
)
from .serialization import stored_document_bytes

_ATTEMPT_TERMINALS = {"succeeded", "failed", "interrupted", "cancelled"}
_TASK_TERMINALS = {"succeeded", "failed", "interrupted", "cancelled", "skipped"}


class ReplayState(Protocol):
    study_state: StudyState
    execution_id: str | None
    updated_at: str
    task_states: dict[str, TaskState]
    attempt_states: dict[str, AttemptState]
    attempts: dict[str, AttemptRecord]


@dataclass(frozen=True, slots=True)
class EffectiveStudyState:
    manifest: StudyManifest
    tasks: tuple[TaskRecord, ...]
    attempts: tuple[AttemptRecord, ...]


def validate_pristine_created_state(
    manifest: StudyManifest,
    tasks: tuple[TaskRecord, ...],
    attempts: tuple[AttemptRecord, ...],
) -> None:
    pristine_tasks = all(
        task.state == "pending"
        and not task.attempts
        and task.claim_epoch == 0
        and task.current_attempt_id is None
        and task.selected_success_attempt_id is None
        and task.retryability.attempts_remaining == manifest.max_attempts_per_task
        and not task.retryability.retryable
        and task.retryability.category is None
        and task.reason is None
        for task in tasks
    )
    if manifest.state != "created" or manifest.execution_id is not None or attempts or not pristine_tasks:
        _integrity(
            "INCONSISTENT_INITIAL_STATE",
            "study_manifest",
            "canonical created state with pristine tasks and no attempts",
            {"state": manifest.state, "execution_id": manifest.execution_id, "attempts": len(attempts)},
        )


def project_effective_study(
    root: Path,
    manifest: StudyManifest,
    tasks: tuple[TaskRecord, ...],
    attempts: tuple[AttemptRecord, ...],
    events: tuple[StudyEvent, ...],
    state: ReplayState,
    checkpoint_state: ReplayState,
) -> EffectiveStudyState:
    validate_root_checkpoint(manifest, checkpoint_state)
    validate_record_checkpoints(tasks, attempts, state)
    effective_attempts = tuple(state.attempts[item.attempt_id] for item in attempts)
    effective_tasks = tuple(_effective_task(item, state, attempts, effective_attempts) for item in tasks)
    effective_manifest = replace(
        manifest,
        state=state.study_state,
        updated_at=state.updated_at,
        execution_id=state.execution_id,
        counts=counts(state.task_states),
        checkpoint_sequence=events[-1].sequence,
        checkpoint_event_sha256=events[-1].file_sha256,
    )
    return EffectiveStudyState(effective_manifest, effective_tasks, effective_attempts)


def validate_root_checkpoint(manifest: StudyManifest, state: ReplayState) -> None:
    expected = {
        "state": state.study_state,
        "execution_id": state.execution_id,
        "updated_at": state.updated_at,
        "counts": counts(state.task_states),
    }
    actual = {
        "state": manifest.state,
        "execution_id": manifest.execution_id,
        "updated_at": manifest.updated_at,
        "counts": manifest.counts,
    }
    if actual != expected:
        raise StudyCheckpointError(
            operation="load study",
            reason="CHECKPOINT_JOURNAL_INCONSISTENCY",
            document_role="study_manifest",
            expected=expected,
            actual=actual,
            action="Restore checkpoints from the matching event prefix; loading does not repair them.",
        )


def validate_record_checkpoints(tasks: tuple[TaskRecord, ...], attempts: tuple[AttemptRecord, ...], state: ReplayState) -> None:
    for task in tasks:
        if not _is_lagging_or_equal(task.state, state.task_states[task.task_id], terminal=_TASK_TERMINALS):
            _checkpoint_ahead("study_task", task.task_id, task.state, state.task_states[task.task_id])
    for attempt in attempts:
        effective = state.attempt_states[attempt.attempt_id]
        if not _is_lagging_or_equal(attempt.status, effective, terminal=_ATTEMPT_TERMINALS):
            _checkpoint_ahead("study_attempt", attempt.attempt_id, attempt.status, effective)


def counts(states: Mapping[str, TaskState]) -> StudyCounts:
    derived = Counter(states.values())
    return StudyCounts(
        tasks=len(states),
        pending=derived["pending"],
        running=derived["running"],
        succeeded=derived["succeeded"],
        failed=derived["failed"],
        interrupted=derived["interrupted"],
        cancelled=derived["cancelled"],
        skipped=derived["skipped"],
    )


def _is_lagging_or_equal(checkpoint: str, effective: str, *, terminal: set[str]) -> bool:
    if checkpoint == effective:
        return True
    if checkpoint in terminal:
        return False
    return checkpoint == "created" or checkpoint == "pending" or (checkpoint == "running" and effective in terminal)


def _effective_task(
    task: TaskRecord,
    state: ReplayState,
    checkpoint_attempts: tuple[AttemptRecord, ...],
    effective_attempts: tuple[AttemptRecord, ...],
) -> TaskRecord:
    task_attempts = tuple(item for item in effective_attempts if item.task_id == task.task_id)
    checkpoint_task_attempts = tuple(item for item in checkpoint_attempts if item.task_id == task.task_id)
    references = tuple(_attempt_reference(item) for item in checkpoint_task_attempts)
    effective_state = state.task_states[task.task_id]
    selected = next((item.attempt_id for item in task_attempts if item.status == "succeeded"), None)
    current = next((item.attempt_id for item in task_attempts if item.status in {"created", "running"}), None)
    retryable, category = _effective_retryability(task, task_attempts, effective_state)
    return replace(
        task,
        state=effective_state,
        attempts=references,
        current_attempt_id=current if effective_state == "running" else None,
        selected_success_attempt_id=selected,
        claim_epoch=max(task.claim_epoch, len(task_attempts)),
        retryability=replace(
            task.retryability,
            retryable=retryable,
            category=category,
            attempts_remaining=max(0, task.retryability.attempts_remaining - max(0, len(task_attempts) - len(task.attempts))),
        ),
    )


def _effective_retryability(
    task: TaskRecord,
    attempts: tuple[AttemptRecord, ...],
    state: TaskState,
) -> tuple[bool, str | None]:
    remaining = max(0, task.retryability.attempts_remaining - max(0, len(attempts) - len(task.attempts)))
    if state == "interrupted":
        return remaining > 0, "interruption"
    if state != "failed" or not attempts:
        return False, None
    failure = attempts[-1].failure
    permitted = bool(failure.get("retryable")) if isinstance(failure, Mapping) else False
    category = failure.get("category") if isinstance(failure, Mapping) else None
    return permitted and remaining > 0, category if isinstance(category, str) else None


def _attempt_reference(attempt: AttemptRecord) -> AttemptReference:
    relative = f"tasks/{attempt.task_id.removeprefix('sha256:')}/attempts/{attempt.attempt_id}.json"
    payload = stored_document_bytes(attempt_document(attempt))
    return AttemptReference(
        attempt_id=attempt.attempt_id,
        attempt_number=attempt.attempt_number,
        path=relative,
        role="attempt",
        required_for=("inspect", "resume"),
        semantic_sha256=attempt.document_sha256,
        sha256=sha256_bytes(payload),
        bytes=len(payload),
    )


def _checkpoint_ahead(role: str, entity_id: str, checkpoint: str, journal: str) -> None:
    raise StudyCheckpointError(
        operation="load study",
        reason="CHECKPOINT_AHEAD_OF_JOURNAL",
        document_role=role,
        entity_id=entity_id,
        expected=f"state no newer than journal state {journal!r}",
        actual=checkpoint,
        action="Restore the checkpoint matching or lagging the immutable journal.",
    )


def _integrity(reason: str, role: str, expected: object, actual: object) -> None:
    raise StudyIntegrityError(
        operation="load study",
        reason=reason,
        document_role=role,
        expected=expected,
        actual=actual,
        action="Restore the contiguous authoritative journal and matching checkpoints.",
    )


__all__ = ["EffectiveStudyState", "project_effective_study", "validate_pristine_created_state"]
