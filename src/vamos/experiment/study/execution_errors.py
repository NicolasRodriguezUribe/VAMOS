"""Context-rich public errors for durable study execution."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .errors import StudyError, StudyFinalizationError, UnsupportedStudyExecutionStateError
from .models import Study, StudyManifest, TaskRecord


class ExecutionContext(Protocol):
    root: Path
    study_id: str
    manifest: StudyManifest
    active_task_id: str | None
    active_attempt_id: str | None
    active_run_id: str | None
    objective_evaluation_began: bool


def state_error(study: Study, reason: str, state: str) -> UnsupportedStudyExecutionStateError:
    return UnsupportedStudyExecutionStateError(
        operation="run study",
        reason=reason,
        study_id=study.study_id,
        current_state=state,
        expected_state="created",
        expected="newly created canonical study",
        actual=state,
        action="Resume and retry are not implemented; inspect this root or create a new study.",
    )


def state_error_for_task(state: ExecutionContext, task: TaskRecord, reason: str) -> UnsupportedStudyExecutionStateError:
    return UnsupportedStudyExecutionStateError(
        operation="select runnable study task",
        reason=reason,
        study_id=state.study_id,
        task_id=task.task_id,
        current_state=task.state,
        expected_state="pending, failed, or interrupted",
        expected="canonical task eligible for this explicit operation",
        actual=task.state,
        action="Reload the study and select only contract-eligible unfinished work.",
    )


def enrich_execution_error(exc: StudyError, snapshot: Study, state: ExecutionContext | None) -> None:
    if exc.study_id is None:
        exc.study_id = snapshot.study_id
    if state is None:
        return
    if exc.task_id is None:
        exc.task_id = state.active_task_id
    if exc.attempt_id is None:
        exc.attempt_id = state.active_attempt_id
    if exc.current_state is None:
        exc.current_state = state.manifest.state
    if exc.expected_state is None:
        exc.expected_state = "completed"
    exc.objective_evaluation_began = exc.objective_evaluation_began or state.objective_evaluation_began
    exc.execution_occurred = exc.objective_evaluation_began
    exc.canonical_run_published = exc.canonical_run_published or active_run_published(state)


def active_run_published(state: ExecutionContext | None) -> bool:
    return state is not None and state.active_run_id is not None and (state.root / "runs" / state.active_run_id).is_dir()


def finalization_error(
    study_id: str,
    current_state: str,
    objective_began: bool,
    run_published: bool,
    root: Path,
    exc: Exception,
) -> StudyFinalizationError:
    return StudyFinalizationError(
        operation="finalize study execution",
        reason="STUDY_FINALIZATION_FAILED",
        study_id=study_id,
        current_state=current_state,
        expected_state="completed",
        objective_evaluation_began=objective_began,
        canonical_run_published=run_published,
        path=root,
        expected="terminal completion event and matching root checkpoint",
        actual=type(exc).__name__,
        action="Load the authoritative journal to inspect the explicit non-repaired state; resume is not implemented.",
    )


__all__ = ["active_run_published", "enrich_execution_error", "finalization_error", "state_error", "state_error_for_task"]
