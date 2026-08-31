"""Canonical run publication, verification, and durable task failure."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, NoReturn, Protocol

from vamos.experiment.artifacts.jsonio import canonical_json_bytes
from vamos.experiment.artifacts.models import StoredRun, deep_freeze, deep_thaw
from vamos.experiment.artifacts.persistence import load_run, save_failed_execution, save_result
from vamos.experiment.artifacts.verification import verify_run

from .commits import append_event, checkpoint_attempt, checkpoint_manifest, checkpoint_task, now_utc, run_reference
from .errors import StudyExecutionError, StudyRunPublicationError, StudyRunVerificationError
from .models import AttemptRecord, PlanTask, StudyEvent, StudyManifest, TaskRecord


class ExecutionState(Protocol):
    root: Path
    study_id: str
    execution_id: str
    manifest: StudyManifest
    tasks: list[TaskRecord]
    event: StudyEvent


def attach_run_metadata(
    result: Any,
    task: PlanTask,
    run_id: str,
    started_at: str,
    completed_at: str,
    runtime_ms: float,
) -> None:
    result.meta["run_artifact_requested_spec"] = deep_thaw(task.requested_run)
    result.meta["run_artifact_resolved_spec"] = deep_thaw(task.resolved_run_spec)
    result.meta["run_artifact_timestamps"] = {
        "started_at": started_at,
        "completed_at": completed_at,
        "runtime_ms": runtime_ms,
    }
    result.meta["run_artifact_entry_point"] = {
        "kind": "python_api",
        "python": {"callable": "Study.run", "arguments_source": "resolved_spec"},
    }
    result.meta["run_artifact_run_id"] = run_id


def publish_success(
    state: ExecutionState,
    task: TaskRecord,
    plan_task: PlanTask,
    run_id: str,
    result: Any,
    phase: Callable[[str], None],
) -> StoredRun:
    """Publish and fully verify one success before it can enter the journal."""
    run_root = state.root / "runs" / run_id
    try:
        save_result(result, run_root)
    except Exception as exc:
        raise StudyRunPublicationError(
            operation="publish study task run",
            reason="RUN_PUBLICATION_FAILED",
            study_id=state.study_id,
            task_id=task.task_id,
            attempt_id=task.current_attempt_id,
            current_state="running",
            expected_state="succeeded",
            objective_evaluation_began=True,
            canonical_run_published=os.path.lexists(run_root),
            path=run_root,
            expected="atomically published canonical run",
            actual=type(exc).__name__,
            action="Inspect the unreferenced run path; no success event was published.",
        ) from exc
    phase("after_canonical_run_publication")
    try:
        report = verify_run(run_root)
        verified = load_run(run_root, verify="all")
        _validate_published_run(verified, report.numerical_bundle_safety, plan_task, run_id)
    except Exception as exc:
        raise StudyRunVerificationError(
            operation="verify study task run",
            reason="RUN_VERIFICATION_FAILED",
            study_id=state.study_id,
            task_id=task.task_id,
            attempt_id=task.current_attempt_id,
            current_state="running",
            expected_state="succeeded",
            objective_evaluation_began=True,
            canonical_run_published=True,
            path=run_root,
            expected="fully verified successful canonical run",
            actual=type(exc).__name__,
            action="Inspect the unreferenced run; no success event or run reference was published.",
        ) from exc
    phase("after_run_verification")
    return verified


def commit_execution_failure(
    state: ExecutionState,
    index: int,
    task: TaskRecord,
    attempt: AttemptRecord,
    plan_task: PlanTask,
    run_id: str,
    exc: Exception,
    *,
    started_at: str,
    objective_began: bool,
) -> NoReturn:
    """Persist a failed run and terminal study state, then raise typed evidence."""
    completed_at = now_utc()
    failure = _attempt_failure(exc, objective_began)
    run_root = state.root / "runs" / run_id
    try:
        stored = save_failed_execution(
            run_root,
            run_id=run_id,
            requested_spec=plan_task.requested_run,
            resolved_spec=plan_task.resolved_run_spec,
            failure={
                "phase": "optimization" if objective_began else "reconstruction",
                "exception_type": type(exc).__name__,
                "message": failure["message"],
                "traceback": None,
                "optimization_executed": objective_began,
            },
            outcome=_failed_outcome(plan_task, objective_began),
            timestamps={"started_at": started_at, "completed_at": completed_at},
        )
        verify_run(run_root)
        stored = load_run(run_root, verify="all")
        if stored.status != "failed" or stored.manifest.task_id != task.task_id:
            raise ValueError("failed run identity or status differs")
        reference = run_reference(stored, study_root=state.root)
    except Exception as publication_error:
        raise StudyRunPublicationError(
            operation="publish failed study task run",
            reason="RUN_PUBLICATION_INTERRUPTED",
            study_id=state.study_id,
            task_id=task.task_id,
            attempt_id=attempt.attempt_id,
            current_state="running",
            expected_state="failed",
            objective_evaluation_began=objective_began,
            canonical_run_published=os.path.lexists(run_root),
            path=run_root,
            expected="verified failed canonical run before failure event",
            actual=type(publication_error).__name__,
            action="Inspect the unreferenced path; no terminal task outcome was fabricated.",
        ) from publication_error
    state.event = append_event(
        state.root,
        state.event,
        event_type="attempt_failed",
        entity_kind="attempt",
        entity_id=attempt.attempt_id,
        transition_from="running",
        transition_to="failed",
        execution_id=state.execution_id,
        reason=failure,
        payload={"task_id": task.task_id, "run_reference": reference, "failure": failure},
    )
    timestamps = dict(attempt.timestamps)
    timestamps["completed_at"] = state.event.timestamp
    attempt = replace(
        attempt,
        status="failed",
        timestamps=deep_freeze(timestamps),
        failure=deep_freeze(failure),
        run_reference=reference,
    )
    attempt, attempt_ref = checkpoint_attempt(state.root, attempt)
    task = replace(
        task,
        state="failed",
        attempts=(*task.attempts[:-1], attempt_ref),
        current_attempt_id=None,
        selected_success_attempt_id=None,
        retryability=replace(task.retryability, retryable=False, category="execution"),
        reason=deep_freeze(failure),
    )
    task = checkpoint_task(state.root, task)
    state.tasks[index] = task
    state.event = append_event(
        state.root,
        state.event,
        event_type="study_failed",
        entity_kind="study",
        entity_id=state.study_id,
        transition_from="running",
        transition_to="failed",
        execution_id=state.execution_id,
        reason=failure,
    )
    state.manifest = checkpoint_manifest(
        state.root,
        state.manifest,
        state="failed",
        execution_id=state.execution_id,
        tasks=tuple(state.tasks),
        event=state.event,
    )
    raise StudyExecutionError(
        operation="execute study task",
        reason="TASK_EXECUTION_FAILED",
        study_id=state.study_id,
        task_id=task.task_id,
        attempt_id=attempt.attempt_id,
        current_state="failed",
        expected_state="succeeded",
        objective_evaluation_began=objective_began,
        canonical_run_published=True,
        expected="successful task execution",
        actual=failure,
        action="Inspect the durable failed run and attempt; no later task executed and retry is not implemented.",
    ) from exc


def _validate_published_run(stored: StoredRun, bundle_status: str, task: PlanTask, run_id: str) -> None:
    arrays = stored.manifest.artifact("result_bundle")
    contract = arrays.array_contract if arrays is not None else None
    valid = (
        stored.status == "succeeded"
        and stored.manifest.run_id == run_id
        and stored.manifest.task_id == task.task_id
        and canonical_json_bytes(stored.manifest.resolved_spec) == canonical_json_bytes(task.resolved_run_spec)
        and bundle_status == "valid"
        and isinstance(contract, Mapping)
        and {"F", "X"}.issubset(contract)
    )
    if not valid:
        raise ValueError("published run identity, status, resolved spec, or required arrays differ")


def _attempt_failure(exc: Exception, objective_began: bool) -> dict[str, Any]:
    del exc
    phase = "objective evaluation" if objective_began else "resolved-spec reconstruction"
    return {
        "category": "execution",
        "code": "OBJECTIVE_EXECUTION_FAILED" if objective_began else "RESOLVED_SPEC_RECONSTRUCTION_FAILED",
        "message": f"Built-in task {phase} failed; inspect the chained exception in the current process.",
        "retryable": False,
        "safe_action": "Inspect the durable failure and create a new study after correcting the cause.",
    }


def _failed_outcome(task: PlanTask, objective_began: bool) -> dict[str, Any]:
    problem = task.resolved_run_spec.get("problem")
    config = problem.get("config") if isinstance(problem, Mapping) else {}
    algorithm = task.resolved_run_spec.get("algorithm")
    algorithm_config = algorithm.get("config") if isinstance(algorithm, Mapping) else {}
    result_mode = algorithm_config.get("result_mode", "unspecified") if isinstance(algorithm_config, Mapping) else "unspecified"
    return {
        "evaluations": None,
        "generations": None,
        "runtime_ms": 0.0,
        "termination_reason": "task_execution_error",
        "result_mode": result_mode,
        "interrupted": objective_began,
        "usable_result": False,
        "n_solutions": None,
        "n_objectives": config.get("n_obj") if isinstance(config, Mapping) else None,
        "n_variables": config.get("n_var") if isinstance(config, Mapping) else None,
        "metrics": {},
    }


__all__ = ["ExecutionState", "attach_run_metadata", "commit_execution_failure", "publish_success"]
