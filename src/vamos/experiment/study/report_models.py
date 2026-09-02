"""Immutable JSON-ready projections of canonical StudyManifest state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vamos.experiment.artifacts.models import deep_thaw


@dataclass(frozen=True, slots=True)
class StudyIssue:
    """One stable, actionable inspection diagnostic."""

    category: str
    reason: str
    message: str
    action: str
    task_id: str | None
    attempt_id: str | None
    run_id: str | None
    path: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "reason": self.reason,
            "message": self.message,
            "action": self.action,
            "task_id": self.task_id,
            "attempt_id": self.attempt_id,
            "run_id": self.run_id,
            "path": self.path,
        }


@dataclass(frozen=True, slots=True)
class StudyAttemptReport:
    """Current durable state and evidence for one immutable attempt."""

    task_id: str
    attempt_id: str
    attempt_number: int
    status: str
    started_at: str | None
    completed_at: str | None
    run_id: str | None
    run_manifest_path: str | None
    run_status: str | None
    run_metadata_verified: bool
    failure: Mapping[str, Any] | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "attempt_id": self.attempt_id,
            "attempt_number": self.attempt_number,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "run_id": self.run_id,
            "run_manifest_path": self.run_manifest_path,
            "run_status": self.run_status,
            "run_metadata_verified": self.run_metadata_verified,
            "failure": deep_thaw(self.failure) if self.failure is not None else None,
        }


@dataclass(frozen=True, slots=True)
class StudyReport:
    """Deterministic current-state report for one canonical study."""

    study_id: str
    plan_id: str
    state: str
    on_error: str
    max_attempts_per_task: int
    created_at: str
    updated_at: str
    counts: Mapping[str, int]
    attempts: tuple[StudyAttemptReport, ...]
    total_attempt_count: int
    verified_run_count: int
    event_head_sequence: int
    event_head_sha256: str
    checkpoint_sequence: int
    checkpoint_event_sha256: str
    journal_checkpoint_relation: str
    reconciliation_required: bool
    runnable_task_ids: tuple[str, ...]
    retryable_task_ids: tuple[str, ...]
    runnable_work: bool
    retryable_failed_work: bool
    changed: bool
    issues: tuple[StudyIssue, ...]
    next_actions: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_type": "vamos.study-report",
            "schema_version": "1.0.0",
            "study_id": self.study_id,
            "plan_id": self.plan_id,
            "state": self.state,
            "policy": {
                "on_error": self.on_error,
                "max_attempts_per_task": self.max_attempts_per_task,
            },
            "timestamps": {"created_at": self.created_at, "updated_at": self.updated_at},
            "counts": dict(self.counts),
            "attempts": [item.as_dict() for item in self.attempts],
            "total_attempt_count": self.total_attempt_count,
            "verified_run_count": self.verified_run_count,
            "journal": {
                "head_sequence": self.event_head_sequence,
                "head_sha256": self.event_head_sha256,
            },
            "checkpoint": {
                "sequence": self.checkpoint_sequence,
                "event_sha256": self.checkpoint_event_sha256,
                "relation": self.journal_checkpoint_relation,
                "reconciliation_required": self.reconciliation_required,
            },
            "runnable_task_ids": list(self.runnable_task_ids),
            "retryable_task_ids": list(self.retryable_task_ids),
            "runnable_work": self.runnable_work,
            "retryable_failed_work": self.retryable_failed_work,
            "changed": self.changed,
            "issues": [item.as_dict() for item in self.issues],
            "next_actions": list(self.next_actions),
        }


@dataclass(frozen=True, slots=True)
class StudySummaryRow:
    """One deterministic task row derived only from canonical records."""

    study_id: str
    plan_id: str
    task_id: str
    plan_index: int
    state: str
    problem_id: str | None
    algorithm_id: str | None
    backend_id: str | None
    seed: int | None
    evaluation_budget: int | None
    population_size: int | None
    attempt_count: int
    latest_attempt_id: str | None
    selected_attempt_id: str | None
    selected_run_id: str | None
    evidence_run_id: str | None
    run_manifest_path: str | None
    run_metadata_available: bool
    run_status: str | None
    evaluations: int | None
    termination_reason: str | None
    runtime_ms: float | int | None
    failure_category: str | None
    failure_code: str | None
    failure: Mapping[str, Any] | None
    retryable: bool
    metrics: Mapping[str, Any] | None
    run_manifest_sha256: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "plan_id": self.plan_id,
            "task_id": self.task_id,
            "plan_index": self.plan_index,
            "state": self.state,
            "problem_id": self.problem_id,
            "algorithm_id": self.algorithm_id,
            "backend_id": self.backend_id,
            "seed": self.seed,
            "evaluation_budget": self.evaluation_budget,
            "population_size": self.population_size,
            "attempt_count": self.attempt_count,
            "latest_attempt_id": self.latest_attempt_id,
            "selected_attempt_id": self.selected_attempt_id,
            "selected_run_id": self.selected_run_id,
            "evidence_run_id": self.evidence_run_id,
            "run_manifest_path": self.run_manifest_path,
            "run_metadata_available": self.run_metadata_available,
            "run_status": self.run_status,
            "evaluations": self.evaluations,
            "termination_reason": self.termination_reason,
            "runtime_ms": self.runtime_ms,
            "failure_category": self.failure_category,
            "failure_code": self.failure_code,
            "failure": deep_thaw(self.failure) if self.failure is not None else None,
            "retryable": self.retryable,
            "metrics": deep_thaw(self.metrics) if self.metrics is not None else None,
            "run_manifest_sha256": self.run_manifest_sha256,
        }


@dataclass(frozen=True, slots=True)
class StudySummary:
    """Deterministic, derived, in-memory summary of every planned task."""

    study_id: str
    plan_id: str
    state: str
    generated_at: str
    root_manifest_sha256: str
    event_head_sequence: int
    event_head_sha256: str
    rows: tuple[StudySummaryRow, ...]
    issues: tuple[StudyIssue, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_type": "vamos.study-summary",
            "schema_version": "1.0.0",
            "study_id": self.study_id,
            "plan_id": self.plan_id,
            "state": self.state,
            "generated_at": self.generated_at,
            "root_manifest_sha256": self.root_manifest_sha256,
            "event_head": {
                "sequence": self.event_head_sequence,
                "sha256": self.event_head_sha256,
            },
            "rows": [item.as_dict() for item in self.rows],
            "issues": [item.as_dict() for item in self.issues],
        }


__all__ = ["StudyAttemptReport", "StudyIssue", "StudyReport", "StudySummary", "StudySummaryRow"]
