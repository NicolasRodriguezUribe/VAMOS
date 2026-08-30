"""Relocatable, confined path handling for v1 run artifacts."""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import NoReturn

from .errors import UnsafeArtifactPathError

_ENCODED_SEPARATOR_PATTERN = r"%(?:2f|5c)"


def validate_relative_artifact_path(value: object, *, role: str, operation: str) -> str:
    """Validate and return a normalized POSIX relative artifact path."""
    if not isinstance(value, str) or not value:
        _raise_unsafe(value, role=role, operation=operation, reason="is not a non-empty string")
    path = value
    if "\x00" in path:
        _raise_unsafe(path, role=role, operation=operation, reason="contains a NUL byte")
    if "\\" in path:
        _raise_unsafe(path, role=role, operation=operation, reason="uses an ambiguous backslash separator")
    if re.search(_ENCODED_SEPARATOR_PATTERN, path, re.IGNORECASE):
        _raise_unsafe(path, role=role, operation=operation, reason="contains an encoded path separator")
    if unicodedata.normalize("NFC", path) != path:
        _raise_unsafe(path, role=role, operation=operation, reason="is not Unicode NFC-normalized")
    windows = PureWindowsPath(path)
    posix = PurePosixPath(path)
    if posix.is_absolute() or windows.is_absolute() or windows.drive or path.startswith(("//", "\\\\")):
        _raise_unsafe(path, role=role, operation=operation, reason="is absolute or drive-qualified")
    if any(part in {"", ".", ".."} for part in path.split("/")):
        _raise_unsafe(path, role=role, operation=operation, reason="contains an empty, current, or parent segment")
    if any(":" in part for part in posix.parts):
        _raise_unsafe(path, role=role, operation=operation, reason="contains a drive or stream separator")
    return posix.as_posix()


def confined_artifact_path(root: Path, relative: str, *, role: str, operation: str, must_exist: bool) -> Path:
    """Resolve an artifact and prove that its effective target stays in ``root``."""
    normalized = validate_relative_artifact_path(relative, role=role, operation=operation)
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise UnsafeArtifactPathError(
            operation=operation,
            artifact_role=role,
            path=root,
            reason="run root cannot be resolved",
            expected="existing readable run directory",
            actual=type(exc).__name__,
            action="Restore or select the complete run directory.",
        ) from exc
    candidate = resolved_root.joinpath(*PurePosixPath(normalized).parts)
    try:
        resolved_candidate = candidate.resolve(strict=must_exist)
    except OSError as exc:
        if must_exist and not candidate.exists():
            return candidate
        raise UnsafeArtifactPathError(
            operation=operation,
            artifact_role=role,
            path=relative,
            reason="cannot be safely resolved",
            expected=f"artifact confined under {resolved_root}",
            actual=type(exc).__name__,
            action="Remove unsafe links or restore the original artifact.",
        ) from exc
    try:
        common = Path(os.path.commonpath((resolved_root, resolved_candidate)))
    except ValueError as exc:
        raise UnsafeArtifactPathError(
            operation=operation,
            artifact_role=role,
            path=relative,
            reason="resolves to a different drive or filesystem root",
            expected=f"artifact confined under {resolved_root}",
            actual=str(resolved_candidate),
            action="Use only relative artifacts stored inside the run directory.",
        ) from exc
    if common != resolved_root:
        raise UnsafeArtifactPathError(
            operation=operation,
            artifact_role=role,
            path=relative,
            reason="resolves outside the run directory",
            expected=f"artifact confined under {resolved_root}",
            actual=str(resolved_candidate),
            action="Remove the escaping link/path and restore a contained artifact.",
        )
    _reject_escaping_reparse_points(resolved_root, candidate, role=role, operation=operation)
    return candidate


def _reject_escaping_reparse_points(root: Path, candidate: Path, *, role: str, operation: str) -> None:
    current = root
    for part in candidate.relative_to(root).parts:
        current = current / part
        if not current.exists() and not current.is_symlink():
            break
        is_junction_method = getattr(current, "is_junction", None)
        is_junction = bool(is_junction_method()) if callable(is_junction_method) else False
        if current.is_symlink() or is_junction:
            try:
                target = current.resolve(strict=True)
                common = Path(os.path.commonpath((root, target)))
            except (OSError, ValueError) as exc:
                raise UnsafeArtifactPathError(
                    operation=operation,
                    artifact_role=role,
                    path=str(candidate.relative_to(root).as_posix()),
                    reason="contains an unresolvable symbolic link or junction",
                    expected=f"link target confined under {root}",
                    actual=type(exc).__name__,
                    action="Remove the suspicious link and restore a regular contained artifact.",
                ) from exc
            if common != root:
                raise UnsafeArtifactPathError(
                    operation=operation,
                    artifact_role=role,
                    path=str(candidate.relative_to(root).as_posix()),
                    reason="escapes through a symbolic link or junction",
                    expected=f"link target confined under {root}",
                    actual=str(target),
                    action="Remove the escaping link and restore a regular contained artifact.",
                )


def _raise_unsafe(value: object, *, role: str, operation: str, reason: str) -> NoReturn:
    raise UnsafeArtifactPathError(
        operation=operation,
        artifact_role=role,
        path=str(value),
        reason=reason,
        expected="NFC-normalized POSIX relative path confined to the run directory",
        actual=value,
        action="Use a relative path with forward slashes and no '.', '..', drive, UNC, NUL, or encoded separators.",
    )


__all__ = ["confined_artifact_path", "validate_relative_artifact_path"]
