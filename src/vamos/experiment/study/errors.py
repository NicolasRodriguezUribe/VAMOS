"""Typed, actionable errors for StudyManifest v1."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class StudyError(Exception):
    """Base error with stable fields suitable for future command rendering."""

    category = "study"

    def __init__(
        self,
        *,
        operation: str,
        reason: str,
        expected: Any,
        actual: Any,
        action: str,
        document_role: str | None = None,
        field: str | None = None,
        path: str | Path | None = None,
        entity_id: str | None = None,
        published: bool = False,
    ) -> None:
        self.operation = operation
        self.reason = reason
        self.expected = expected
        self.actual = actual
        self.action = action
        self.document_role = document_role
        self.field = field
        self.path = str(path) if path is not None else None
        self.entity_id = entity_id
        self.published = published
        self.execution_occurred = False
        location = field or document_role or entity_id or "study"
        if self.path is not None:
            location = f"{location} at {self.path}"
        super().__init__(f"Cannot {operation}: {location} ({reason}). Expected {expected!r}; received {actual!r}. {action}")

    def as_dict(self) -> dict[str, Any]:
        """Return stable structured diagnostics."""
        return {
            "operation": self.operation,
            "category": self.category,
            "reason": self.reason,
            "expected": self.expected,
            "actual": self.actual,
            "action": self.action,
            "document_role": self.document_role,
            "field": self.field,
            "path": self.path,
            "entity_id": self.entity_id,
            "published": self.published,
            "execution_occurred": self.execution_occurred,
        }


class InvalidStudySpecError(StudyError):
    category = "study_spec"


class UnresolvedStudyTaskError(StudyError):
    category = "study_resolution"


class DuplicateStudyTaskError(UnresolvedStudyTaskError):
    category = "duplicate_study_task"


class MalformedStudyError(StudyError):
    category = "study_schema"


class UnsupportedStudySchemaError(MalformedStudyError):
    category = "unsupported_study_schema"


class MissingStudyDocumentError(MalformedStudyError):
    category = "missing_study_document"


class StudyIntegrityError(MalformedStudyError):
    category = "study_integrity"


class PlanMismatchError(StudyIntegrityError):
    category = "plan_mismatch"


class UnsafeStudyPathError(MalformedStudyError):
    category = "unsafe_study_path"


class StudyResourceLimitError(MalformedStudyError):
    category = "study_resource_limit"


class StudyOutputCollisionError(StudyError):
    category = "study_output_collision"


class StudyInfrastructureError(StudyError):
    category = "study_infrastructure"


__all__ = [
    "DuplicateStudyTaskError",
    "InvalidStudySpecError",
    "MalformedStudyError",
    "MissingStudyDocumentError",
    "PlanMismatchError",
    "StudyError",
    "StudyInfrastructureError",
    "StudyIntegrityError",
    "StudyOutputCollisionError",
    "StudyResourceLimitError",
    "UnresolvedStudyTaskError",
    "UnsafeStudyPathError",
    "UnsupportedStudySchemaError",
]
