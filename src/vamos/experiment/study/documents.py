"""Canonical initial StudyManifest v1 document construction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vamos.experiment.artifacts.models import deep_thaw

from .models import SCHEMA_VERSION, DocumentReference, PlanTask, ResolvedStudyPlan, StudySpec
from .serialization import seal_document


def study_spec_document(spec: StudySpec, study_id: str) -> dict[str, Any]:
    payload = {
        "document_type": "vamos.study-spec",
        "schema_version": SCHEMA_VERSION,
        "study_id": study_id,
        **spec.as_intent_dict(),
        "integrity": {},
    }
    return seal_document(payload)


def plan_document(plan: ResolvedStudyPlan) -> dict[str, Any]:
    payload = {
        "document_type": "vamos.resolved-study-plan",
        "schema_version": SCHEMA_VERSION,
        "plan_id": plan.plan_id,
        "task_count": plan.task_count,
        "tasks": [_plan_task(task) for task in plan.tasks],
        "integrity": {},
    }
    return seal_document(payload)


def task_document(*, study_id: str, task: PlanTask, max_attempts_per_task: int) -> dict[str, Any]:
    return seal_document(
        {
            "document_type": "vamos.study-task",
            "schema_version": SCHEMA_VERSION,
            "study_id": study_id,
            "task_id": task.task_id,
            "plan_index": task.plan_index,
            "state": "pending",
            "attempts": [],
            "current_attempt_id": None,
            "selected_success_attempt_id": None,
            "retryability": {
                "retryable": False,
                "category": None,
                "attempts_remaining": max_attempts_per_task,
            },
            "reason": None,
            "claim_epoch": 0,
            "integrity": {},
        }
    )


def initial_event_document(*, study_id: str, event_id: str, timestamp: str) -> dict[str, Any]:
    return seal_document(
        {
            "document_type": "vamos.study-event",
            "schema_version": SCHEMA_VERSION,
            "sequence": 1,
            "event_id": event_id,
            "event_type": "study_created",
            "entity": {"kind": "study", "id": study_id},
            "transition": {"from": None, "to": "created"},
            "execution_id": None,
            "timestamp": timestamp,
            "reason": None,
            "payload": {},
            "previous_event_sha256": None,
            "integrity": {},
        }
    )


def manifest_document(
    *,
    study_id: str,
    plan_id: str,
    timestamp: str,
    on_error: str,
    max_attempts_per_task: int,
    task_count: int,
    spec_reference: DocumentReference,
    plan_reference: DocumentReference,
    event_sha256: str,
) -> dict[str, Any]:
    return seal_document(
        {
            "document_type": "vamos.study-manifest",
            "schema_version": SCHEMA_VERSION,
            "study_id": study_id,
            "plan_id": plan_id,
            "state": "created",
            "created_at": timestamp,
            "updated_at": timestamp,
            "execution_id": None,
            "policy": {
                "on_error": on_error,
                "max_attempts_per_task": max_attempts_per_task,
            },
            "spec": reference_document(spec_reference),
            "plan": reference_document(plan_reference),
            "counts": {
                "tasks": task_count,
                "pending": task_count,
                "running": 0,
                "succeeded": 0,
                "failed": 0,
                "interrupted": 0,
                "cancelled": 0,
                "skipped": 0,
            },
            "checkpoint": {"sequence": 1, "event_sha256": event_sha256},
            "integrity": {},
        }
    )


def reference_document(reference: DocumentReference) -> dict[str, Any]:
    return {
        "path": reference.path,
        "role": reference.role,
        "required_for": list(reference.required_for),
        "semantic_sha256": reference.semantic_sha256,
        "sha256": reference.sha256,
        "bytes": reference.bytes,
    }


def semantic_hash(document: Mapping[str, Any]) -> str:
    integrity = document.get("integrity")
    if not isinstance(integrity, Mapping):
        raise AssertionError("sealed document has no integrity object")
    value = integrity.get("document_sha256")
    if not isinstance(value, str):
        raise AssertionError("sealed document has no semantic hash")
    return value


def _plan_task(task: PlanTask) -> dict[str, Any]:
    return {
        "plan_index": task.plan_index,
        "requested_run": deep_thaw(task.requested_run),
        "resolved_run_spec": deep_thaw(task.resolved_run_spec),
        "task_id": task.task_id,
        "task_spec_sha256": task.task_spec_sha256,
    }


__all__ = [
    "initial_event_document",
    "manifest_document",
    "plan_document",
    "reference_document",
    "semantic_hash",
    "study_spec_document",
    "task_document",
]
