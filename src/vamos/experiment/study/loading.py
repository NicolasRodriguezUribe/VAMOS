"""Complete, data-only loading and verification of StudyManifest v1 roots."""

from __future__ import annotations

import hmac
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vamos.experiment.artifacts.jsonio import sha256_bytes

from .decoding import decode_manifest, decode_plan, decode_spec
from .errors import (
    MalformedStudyError,
    MissingStudyDocumentError,
    PlanMismatchError,
    StudyIntegrityError,
    StudyResourceLimitError,
)
from .limits import StudyLoadLimits
from .models import DocumentReference, Study, StudyCounts, StudyEvent, StudyManifest, StudySpec, TaskRecord
from .paths import confined_study_path
from .record_decoding import decode_event, decode_task
from .serialization import load_json, stored_document_bytes


@dataclass(slots=True)
class _ReadBudget:
    limits: StudyLoadLimits
    documents: int = 0
    total_bytes: int = 0

    def account(self, size: int, *, role: str, path: str) -> None:
        self.documents += 1
        self.total_bytes += size
        if self.documents > self.limits.max_documents:
            self._raise("max_documents", self.limits.max_documents, self.documents, role, path)
        if self.total_bytes > self.limits.max_total_bytes:
            self._raise("max_total_bytes", self.limits.max_total_bytes, self.total_bytes, role, path)

    @staticmethod
    def _raise(limit: str, configured: int, observed: int, role: str, path: str) -> None:
        raise StudyResourceLimitError(
            operation="load study",
            reason="RESOURCE_LIMIT",
            document_role=role,
            path=path,
            expected={limit: configured},
            actual=observed,
            action="Inspect the source and pass larger explicit limits only for a trusted study.",
        )


def load_study(path: str | Path, *, limits: StudyLoadLimits | None = None) -> Study:
    """Load and fully verify one canonical study without executing code."""
    configured = limits or StudyLoadLimits()
    root = _root(path)
    budget = _ReadBudget(configured)
    manifest_raw, _ = _read(root, "study-manifest.json", "study_manifest", configured.max_manifest_bytes, budget)
    manifest = decode_manifest(manifest_raw)
    _validate_strings(manifest_raw, configured, "study_manifest")
    _validate_reference_contract(manifest)

    spec_raw, spec_bytes = _read_reference(root, manifest.spec, configured.max_spec_bytes, budget)
    _verify_reference(manifest.spec, spec_raw, spec_bytes)
    spec = decode_spec(spec_raw)

    plan_raw, plan_bytes = _read_reference(root, manifest.plan, configured.max_plan_bytes, budget)
    _verify_reference(manifest.plan, plan_raw, plan_bytes, plan=True)
    plan = decode_plan(plan_raw)
    if plan.task_count > configured.max_tasks:
        raise StudyResourceLimitError(
            operation="load study",
            reason="RESOURCE_LIMIT",
            document_role="resolved_plan",
            field="$.task_count",
            expected=f"at most {configured.max_tasks}",
            actual=plan.task_count,
            action="Split the trusted campaign into smaller studies.",
        )

    tasks = _load_tasks(root, manifest, plan.tasks, configured, budget)
    events = _load_events(root, manifest, configured, budget)
    _cross_validate(manifest, spec, plan.plan_id, tasks, events)
    return Study(root=root, manifest=manifest, spec=spec, plan=plan, tasks=tasks, events=events)


def _root(path: str | Path) -> Path:
    candidate = Path(path)
    try:
        if not candidate.exists() or not candidate.is_dir():
            raise MissingStudyDocumentError(
                operation="load study",
                reason="MISSING_STUDY_ROOT",
                document_role="study_root",
                path=candidate,
                expected="existing canonical StudyManifest v1 directory",
                actual="missing or not a directory",
                action="Select the complete study directory containing study-manifest.json.",
            )
        return candidate.resolve(strict=True)
    except MissingStudyDocumentError:
        raise
    except OSError as exc:
        raise MalformedStudyError(
            operation="load study",
            reason="UNREADABLE_STUDY_ROOT",
            document_role="study_root",
            path=candidate,
            expected="readable study directory",
            actual=type(exc).__name__,
            action="Restore permissions or select a readable canonical study.",
        ) from exc


def _read(
    root: Path,
    relative: str,
    role: str,
    max_bytes: int,
    budget: _ReadBudget,
) -> tuple[dict[str, Any], bytes]:
    target = confined_study_path(root, relative, role=role, must_exist=True)
    if not target.exists() or not target.is_file():
        raise MissingStudyDocumentError(
            operation="load study",
            reason="MISSING_DOCUMENT",
            document_role=role,
            path=relative,
            expected="regular canonical JSON file",
            actual="missing or not a file",
            action="Restore the complete study directory from a trusted copy.",
        )
    try:
        size = target.stat().st_size
    except OSError as exc:
        raise MalformedStudyError(
            operation="load study",
            reason="UNREADABLE_DOCUMENT",
            document_role=role,
            path=relative,
            expected="readable regular file",
            actual=type(exc).__name__,
            action="Restore the canonical document and its permissions.",
        ) from exc
    budget.account(size, role=role, path=relative)
    value = load_json(target, operation="load study", role=role, max_bytes=max_bytes, max_depth=budget.limits.max_json_depth)
    try:
        payload = target.read_bytes()
    except OSError as exc:
        raise MalformedStudyError(
            operation="load study",
            reason="UNREADABLE_DOCUMENT",
            document_role=role,
            path=relative,
            expected="readable canonical bytes",
            actual=type(exc).__name__,
            action="Restore the canonical document and its permissions.",
        ) from exc
    expected_bytes = stored_document_bytes(value)
    if payload != expected_bytes:
        raise StudyIntegrityError(
            operation="load study",
            reason="NON_CANONICAL_BYTES",
            document_role=role,
            path=relative,
            expected=sha256_bytes(expected_bytes),
            actual=sha256_bytes(payload),
            action="Restore the exact canonical JSON bytes; semantic reformatting is not a persisted write.",
        )
    _validate_strings(value, budget.limits, role)
    return value, payload


def _read_reference(
    root: Path,
    reference: DocumentReference,
    max_bytes: int,
    budget: _ReadBudget,
) -> tuple[dict[str, Any], bytes]:
    return _read(root, reference.path, reference.role, max_bytes, budget)


def _verify_reference(
    reference: DocumentReference,
    value: Mapping[str, Any],
    payload: bytes,
    *,
    plan: bool = False,
) -> None:
    file_hash = sha256_bytes(payload)
    integrity = value.get("integrity")
    semantic = integrity.get("document_sha256") if isinstance(integrity, Mapping) else None
    matches = reference.bytes == len(payload) and hmac.compare_digest(reference.sha256, file_hash)
    matches = matches and isinstance(semantic, str) and hmac.compare_digest(reference.semantic_sha256, semantic)
    if not matches:
        error = PlanMismatchError if plan else StudyIntegrityError
        raise error(
            operation="load study",
            reason="PLAN_MISMATCH" if plan else "DOCUMENT_REFERENCE_MISMATCH",
            document_role=reference.role,
            path=reference.path,
            expected={
                "bytes": reference.bytes,
                "sha256": reference.sha256,
                "semantic_sha256": reference.semantic_sha256,
            },
            actual={"bytes": len(payload), "sha256": file_hash, "semantic_sha256": semantic},
            action="Restore the immutable published document and matching root manifest.",
        )


def _load_tasks(
    root: Path,
    manifest: StudyManifest,
    plan_tasks: tuple[Any, ...],
    limits: StudyLoadLimits,
    budget: _ReadBudget,
) -> tuple[TaskRecord, ...]:
    if not plan_tasks:
        _assert_no_task_entries(root)
        return ()
    result: list[TaskRecord] = []
    expected_dirs: set[str] = set()
    for plan_task in plan_tasks:
        digest = plan_task.task_id.removeprefix("sha256:")
        expected_dirs.add(digest)
        relative = f"tasks/{digest}/task.json"
        raw, _ = _read(root, relative, "study_task", limits.max_task_bytes, budget)
        task = decode_task(raw)
        if (task.study_id, task.task_id, task.plan_index) != (
            manifest.study_id,
            plan_task.task_id,
            plan_task.plan_index,
        ):
            raise StudyIntegrityError(
                operation="load study",
                reason="TASK_ID_MISMATCH",
                document_role="study_task",
                path=relative,
                expected=(manifest.study_id, plan_task.task_id, plan_task.plan_index),
                actual=(task.study_id, task.task_id, task.plan_index),
                action="Restore the task checkpoint matching the immutable plan.",
            )
        result.append(task)
    _assert_task_entries(root, expected_dirs, initial=manifest.state == "created")
    return tuple(result)


def _assert_no_task_entries(root: Path) -> None:
    tasks_root = root / "tasks"
    if tasks_root.exists():
        if _entry_names(root, "tasks", "tasks"):
            _unexpected("tasks", "no task entries for an empty plan", "non-empty tasks directory")


def _assert_task_entries(root: Path, expected: set[str], *, initial: bool) -> None:
    observed = _entry_names(root, "tasks", "tasks")
    if observed != expected:
        _unexpected("tasks", sorted(expected), sorted(observed))
    if initial:
        for digest in expected:
            entries = _entry_names(root, f"tasks/{digest}", "study_task")
            if entries != {"task.json"}:
                _unexpected(f"tasks/{digest}", ["task.json"], sorted(entries))


def _load_events(
    root: Path,
    manifest: StudyManifest,
    limits: StudyLoadLimits,
    budget: _ReadBudget,
) -> tuple[StudyEvent, ...]:
    if manifest.checkpoint_sequence > limits.max_documents:
        _ReadBudget._raise("max_documents", limits.max_documents, manifest.checkpoint_sequence, "study_event", "events")
    result: list[StudyEvent] = []
    previous_hash: str | None = None
    for sequence in range(1, manifest.checkpoint_sequence + 1):
        relative = f"events/{sequence:020d}.json"
        raw, payload = _read(root, relative, "study_event", limits.max_event_bytes, budget)
        event = decode_event(raw, file_sha256=sha256_bytes(payload))
        if event.sequence != sequence or event.previous_event_sha256 != previous_hash:
            raise StudyIntegrityError(
                operation="load study",
                reason="EVENT_HASH_CHAIN_BROKEN",
                document_role="study_event",
                path=relative,
                expected={"sequence": sequence, "previous_event_sha256": previous_hash},
                actual={"sequence": event.sequence, "previous_event_sha256": event.previous_event_sha256},
                action="Restore the complete contiguous immutable event journal.",
            )
        result.append(event)
        previous_hash = event.file_sha256
    expected_names = {f"{sequence:020d}.json" for sequence in range(1, manifest.checkpoint_sequence + 1)}
    observed_names = _entry_names(root, "events", "events")
    if observed_names != expected_names:
        _unexpected("events", sorted(expected_names), sorted(observed_names))
    if previous_hash != manifest.checkpoint_event_sha256:
        raise StudyIntegrityError(
            operation="load study",
            reason="EVENT_HEAD_MISMATCH",
            document_role="study_manifest",
            field="$.checkpoint.event_sha256",
            expected=previous_hash,
            actual=manifest.checkpoint_event_sha256,
            action="Restore the root checkpoint matching the immutable event head.",
        )
    return tuple(result)


def _cross_validate(
    manifest: StudyManifest,
    spec: StudySpec,
    plan_id: str,
    tasks: tuple[TaskRecord, ...],
    events: tuple[StudyEvent, ...],
) -> None:
    _validate_manifest_identity(manifest, spec, plan_id)
    _validate_manifest_counts(manifest, tasks)
    _validate_manifest_policy(manifest, spec)
    if manifest.state == "created":
        _validate_initial_state(manifest, tasks, events)


def _validate_manifest_identity(manifest: StudyManifest, spec: StudySpec, plan_id: str) -> None:
    if spec.study_id != manifest.study_id or plan_id != manifest.plan_id:
        raise PlanMismatchError(
            operation="load study",
            reason="PLAN_MISMATCH",
            expected={"study_id": manifest.study_id, "plan_id": manifest.plan_id},
            actual={"study_id": spec.study_id, "plan_id": plan_id},
            action="Restore the matching spec, plan, and manifest set.",
        )


def _validate_manifest_counts(manifest: StudyManifest, tasks: tuple[TaskRecord, ...]) -> None:
    derived = Counter(task.state for task in tasks)
    expected_counts = StudyCounts(
        tasks=len(tasks),
        pending=derived["pending"],
        running=derived["running"],
        succeeded=derived["succeeded"],
        failed=derived["failed"],
        interrupted=derived["interrupted"],
        cancelled=derived["cancelled"],
        skipped=derived["skipped"],
    )
    if manifest.counts != expected_counts:
        _unexpected("study_manifest", expected_counts, manifest.counts, reason="COUNT_MISMATCH")


def _validate_manifest_policy(manifest: StudyManifest, spec: StudySpec) -> None:
    if (manifest.on_error, manifest.max_attempts_per_task) != (
        spec.on_error,
        spec.max_attempts_per_task,
    ):
        _unexpected(
            "study_manifest.policy",
            (spec.on_error, spec.max_attempts_per_task),
            (manifest.on_error, manifest.max_attempts_per_task),
            reason="POLICY_MISMATCH",
        )


def _validate_initial_state(
    manifest: StudyManifest,
    tasks: tuple[TaskRecord, ...],
    events: tuple[StudyEvent, ...],
) -> None:
    valid_initial = (
        manifest.execution_id is None
        and all(_is_initial_task(task, manifest.max_attempts_per_task) for task in tasks)
        and len(events) == 1
        and _is_initial_event(events[0], manifest)
    )
    if not valid_initial:
        _unexpected("study_manifest", "canonical created state", "inconsistent initial checkpoints", reason="INCONSISTENT_INITIAL_STATE")


def _is_initial_task(task: TaskRecord, max_attempts: int) -> bool:
    return (
        task.state == "pending"
        and not task.attempts
        and task.claim_epoch == 0
        and task.current_attempt_id is None
        and task.selected_success_attempt_id is None
        and task.retryability.attempts_remaining == max_attempts
        and not task.retryability.retryable
        and task.retryability.category is None
        and task.reason is None
    )


def _is_initial_event(event: StudyEvent, manifest: StudyManifest) -> bool:
    return (
        event.event_type == "study_created"
        and event.entity_kind == "study"
        and event.entity_id == manifest.study_id
        and event.transition_from is None
        and event.transition_to == "created"
        and event.execution_id is None
        and event.timestamp == manifest.created_at == manifest.updated_at
        and event.reason is None
        and not event.payload
    )


def _validate_reference_contract(manifest: StudyManifest) -> None:
    expected = {
        "spec": ("study-spec.json", "study_spec"),
        "plan": ("plan.json", "resolved_plan"),
    }
    for name, reference in (("spec", manifest.spec), ("plan", manifest.plan)):
        path, role = expected[name]
        if reference.path != path or reference.role != role or reference.required_for != ("inspect", "run", "resume"):
            _unexpected(
                f"study_manifest.{name}",
                {"path": path, "role": role, "required_for": ("inspect", "run", "resume")},
                reference,
            )


def _validate_strings(value: object, limits: StudyLoadLimits, role: str) -> None:
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str) and len(item.encode("utf-8")) > limits.max_string_bytes:
            raise StudyResourceLimitError(
                operation="load study",
                reason="RESOURCE_LIMIT",
                document_role=role,
                expected=f"strings at most {limits.max_string_bytes} UTF-8 bytes",
                actual=len(item.encode("utf-8")),
                action="Remove oversized untrusted metadata.",
            )
        if isinstance(item, Mapping):
            stack.extend(item.keys())
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)


def _entry_names(root: Path, relative: str, role: str) -> set[str]:
    directory = confined_study_path(root, relative, role=role, must_exist=True)
    try:
        if not directory.is_dir():
            raise OSError("not a directory")
        return {entry.name for entry in directory.iterdir()}
    except OSError as exc:
        raise MalformedStudyError(
            operation="load study",
            reason="UNREADABLE_DOCUMENT_DIRECTORY",
            document_role=role,
            path=relative,
            expected="readable contained directory",
            actual=type(exc).__name__,
            action="Restore the canonical directory and its permissions.",
        ) from exc


def _unexpected(role: str, expected: object, actual: object, *, reason: str = "UNEXPECTED_LAYOUT") -> None:
    raise StudyIntegrityError(
        operation="load study",
        reason=reason,
        document_role=role,
        expected=expected,
        actual=actual,
        action="Restore the complete canonical study tree; do not infer state from extra or missing files.",
    )


__all__ = ["load_study"]
