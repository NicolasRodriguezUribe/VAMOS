"""Relocatable, root-confined paths for StudyManifest v1."""

from __future__ import annotations

from pathlib import Path

from vamos.experiment.artifacts.errors import RunArtifactError
from vamos.experiment.artifacts.paths import confined_artifact_path, validate_relative_artifact_path

from .errors import UnsafeStudyPathError


def validate_study_relative_path(value: object, *, role: str, operation: str = "load study") -> str:
    try:
        return validate_relative_artifact_path(value, role=role, operation=operation)
    except RunArtifactError as exc:
        raise _study_path_error(exc, role=role, operation=operation) from exc


def confined_study_path(
    root: Path,
    relative: str,
    *,
    role: str,
    operation: str = "load study",
    must_exist: bool = True,
) -> Path:
    try:
        return confined_artifact_path(root, relative, role=role, operation=operation, must_exist=must_exist)
    except RunArtifactError as exc:
        raise _study_path_error(exc, role=role, operation=operation) from exc


def _study_path_error(exc: RunArtifactError, *, role: str, operation: str) -> UnsafeStudyPathError:
    return UnsafeStudyPathError(
        operation=operation,
        reason="UNSAFE_PATH",
        document_role=role,
        field=exc.field,
        path=exc.path,
        expected="NFC-normalized POSIX relative path confined to the study root",
        actual=exc.actual,
        action="Restore a relative contained path and remove escaping symbolic links or junctions.",
    )


__all__ = ["confined_study_path", "validate_study_relative_path"]
