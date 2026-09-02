"""Immutable command envelope and table-driven durable-study exit policy."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Literal, cast

from vamos.experiment.artifacts.models import deep_freeze, deep_thaw
from vamos.experiment.study.errors import (
    InvalidStudySpecError,
    InvalidStudyTransitionError,
    MalformedStudyError,
    PlanMismatchError,
    ResumeEnvironmentIncompatibilityError,
    RetryNotAllowedError,
    StudyError,
    StudyInfrastructureError,
    StudyOutputCollisionError,
    UnresolvedStudyTaskError,
    UnsupportedStudyExecutionStateError,
    UnsupportedStudySchemaError,
)
from vamos.experiment.study.report_models import StudyReport

COMMAND_DOCUMENT_TYPE = "vamos.study-command-result"
COMMAND_SCHEMA_VERSION = "1.0.0"
Operation = Literal["plan", "create", "run", "inspect", "resume", "retry", "summarize"]


@dataclass(frozen=True, slots=True)
class StudyCommandResult:
    operation: Operation
    status: str
    exit_code: int
    study_id: str | None
    plan_id: str | None
    changed: bool
    payload: Mapping[str, Any]
    warnings: tuple[str, ...]
    errors: tuple[Mapping[str, Any], ...]
    next_actions: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", cast(Mapping[str, Any], deep_freeze(_sanitize(self.payload))))
        frozen_errors = tuple(cast(Mapping[str, Any], deep_freeze(_sanitize(item))) for item in self.errors)
        object.__setattr__(self, "errors", frozen_errors)

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_type": COMMAND_DOCUMENT_TYPE,
            "schema_version": COMMAND_SCHEMA_VERSION,
            "operation": self.operation,
            "status": self.status,
            "exit_code": self.exit_code,
            "study_id": self.study_id,
            "plan_id": self.plan_id,
            "changed": self.changed,
            "payload": deep_thaw(self.payload),
            "warnings": list(self.warnings),
            "errors": [deep_thaw(item) for item in self.errors],
            "next_actions": list(self.next_actions),
        }


def map_exit_code(operation: Operation, outcome: StudyReport | StudyError | str) -> int:
    """Map one typed outcome through the frozen SA-072 exit table."""
    if isinstance(outcome, StudyReport):
        if outcome.issues:
            return 3
        return _STATE_EXIT_CODES.get(outcome.state, 0)
    if isinstance(outcome, StudyError):
        reason_exit = _REASON_EXIT_CODES.get(outcome.reason)
        if reason_exit is not None:
            return reason_exit
        for error_types, exit_code in _ERROR_EXIT_ROWS:
            if isinstance(outcome, error_types):
                return exit_code
        return 7
    return _STATE_EXIT_CODES.get(outcome, 0)


def _sanitize(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_sanitize(item) for item in value]
    if isinstance(value, Path):
        return value.name if value.is_absolute() else value.as_posix()
    if isinstance(value, str) and (os.path.isabs(value) or bool(PureWindowsPath(value).drive)):
        return Path(value).name or "<absolute-path>"
    return value


_STATE_EXIT_CODES = {"blocked": 5, "paused": 6, "completed_with_failures": 6, "cancelled": 8, "interrupted": 8}
_REASON_EXIT_CODES = {"ACTIVE_IN_PROCESS_OWNERSHIP": 5, "ACTIVE_LEASE": 5, "TASK_ALREADY_CLAIMED": 5}
_ERROR_EXIT_ROWS: tuple[tuple[tuple[type[StudyError], ...], int], ...] = (
    ((InvalidStudySpecError, UnresolvedStudyTaskError), 2),
    (
        (
            UnsupportedStudySchemaError,
            PlanMismatchError,
            InvalidStudyTransitionError,
            UnsupportedStudyExecutionStateError,
            RetryNotAllowedError,
        ),
        4,
    ),
    ((StudyOutputCollisionError,), 5),
    ((ResumeEnvironmentIncompatibilityError, StudyInfrastructureError), 7),
    ((MalformedStudyError,), 3),
)


__all__ = [
    "COMMAND_DOCUMENT_TYPE",
    "COMMAND_SCHEMA_VERSION",
    "Operation",
    "StudyCommandResult",
    "map_exit_code",
]
