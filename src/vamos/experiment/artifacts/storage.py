"""Non-destructive atomic directory storage for v1 run artifacts."""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .bundle import array_contract, load_result_bundle, write_result_bundle
from .errors import ArtifactResourceLimitError, OutputCollisionError
from .jsonio import sha256_file, stored_json_bytes
from .manifest import build_terminal_manifest
from .models import ArtifactDescriptor, LoadLimits, RunManifest


def store_succeeded_run(
    destination: Path,
    *,
    arrays: Mapping[str, np.ndarray],
    environment: Mapping[str, Any],
    manifest_base: Mapping[str, Any],
    limits: LoadLimits,
) -> RunManifest:
    """Write a complete run in a sibling staging directory and publish once."""
    destination = destination.absolute()
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    _reject_existing(destination)
    token = uuid.uuid4().hex
    staging = parent / f".{destination.name}.vamos-staging-{token}"
    lock = parent / f".{destination.name}.vamos-save.lock"
    acquired_lock = False
    owns_staging = False
    published = False
    try:
        try:
            with lock.open("xb") as handle:
                handle.write(token.encode("ascii"))
                handle.flush()
                os.fsync(handle.fileno())
            acquired_lock = True
        except FileExistsError as exc:
            raise _collision(destination, "another save currently owns the destination") from exc
        _reject_existing(destination)
        staging.mkdir(exist_ok=False)
        owns_staging = True
        _write_running_manifest(staging, manifest_base)

        result_path = staging / "result.npz"
        result_temp = staging / ".result.npz.tmp"
        write_result_bundle(result_temp, arrays, limits=limits)
        os.replace(result_temp, result_path)
        _check_stored_size(result_path, limit=limits.max_artifact_bytes, limit_name="max_artifact_bytes", role="result_bundle")

        environment_path = staging / "environment.json"
        _write_bytes_atomic(environment_path, stored_json_bytes(environment))
        _check_stored_size(
            environment_path,
            limit=limits.max_environment_bytes,
            limit_name="max_environment_bytes",
            role="environment",
        )
        compatibility_paths = _write_compatibility_views(staging, arrays, manifest_base)
        for role, compatibility_path, _media_type, _required_for in compatibility_paths:
            _check_stored_size(
                compatibility_path,
                limit=limits.max_artifact_bytes,
                limit_name="max_artifact_bytes",
                role=role,
            )
        environment_descriptor = _descriptor(
            "environment",
            environment_path,
            "application/vnd.vamos.environment+json",
            required_for=["inspect", "verify", "replay"],
            canonical=True,
        )
        result_descriptor = {
            **_descriptor(
                "result_bundle",
                result_path,
                "application/vnd.vamos.result-bundle+npz",
                required_for=["load", "inspect", "verify", "replay", "analysis"],
                canonical=True,
            ),
            "array_contract": array_contract(arrays),
        }
        load_result_bundle(
            result_path,
            descriptor=_artifact_descriptor(result_descriptor),
            limits=limits,
            required_f=True,
            operation="save result",
        )
        descriptors = [environment_descriptor, result_descriptor]
        descriptors.extend(
            _descriptor(role, path, media_type, required_for=required_for, canonical=False)
            for role, path, media_type, required_for in compatibility_paths
        )
        terminal_value = dict(manifest_base)
        terminal_value["artifacts"] = descriptors
        terminal = build_terminal_manifest(terminal_value, limits=limits)
        _write_bytes_atomic(staging / "manifest.json", stored_json_bytes(terminal.as_dict()))
        _fsync_directory(staging)
        _reject_existing(destination)
        os.rename(staging, destination)
        published = True
        _fsync_directory(parent)
        return terminal
    except FileExistsError as exc:
        raise _collision(destination, "already exists") from exc
    finally:
        if not published and owns_staging:
            _remove_owned_staging(staging, parent=parent, token=token)
        if acquired_lock:
            try:
                lock.unlink(missing_ok=True)
            except OSError:
                logging.getLogger(__name__).warning("Could not remove owned save lock %s", lock, exc_info=True)


def _write_running_manifest(staging: Path, manifest_base: Mapping[str, Any]) -> None:
    running = dict(manifest_base)
    running["status"] = "running"
    running["artifacts"] = []
    running.pop("outcome", None)
    running.pop("integrity", None)
    timestamps = dict(running.get("timestamps", {}))
    timestamps.pop("completed_at", None)
    running["timestamps"] = timestamps
    _write_bytes_atomic(staging / "manifest.json", stored_json_bytes(running))


def _write_compatibility_views(
    staging: Path,
    arrays: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
) -> list[tuple[str, Path, str, list[str]]]:
    views: list[tuple[str, Path, str, list[str]]] = []
    for key, filename, role in (("F", "FUN.csv", "objectives_csv"), ("X", "X.csv", "decisions_csv")):
        array = arrays.get(key)
        if array is None:
            continue
        path = staging / filename
        temp = staging / f".{filename}.tmp"
        with temp.open("xb") as handle:
            np.savetxt(handle, array, delimiter=",", fmt="%.18e")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        views.append((role, path, "text/csv", ["inspect", "analysis"]))
    outcome = manifest.get("outcome")
    metadata = {
        "document_type": "vamos.compatibility-metadata",
        "schema_version": "1.0.0",
        "run_id": manifest.get("run_id"),
        "task_id": manifest.get("task_id"),
        "n_solutions": outcome.get("n_solutions") if isinstance(outcome, Mapping) else None,
        "n_objectives": outcome.get("n_objectives") if isinstance(outcome, Mapping) else None,
    }
    metadata_path = staging / "metadata.json"
    _write_bytes_atomic(metadata_path, stored_json_bytes(metadata))
    views.append(("metadata_view", metadata_path, "application/json", ["inspect"]))
    return views


def _descriptor(
    role: str,
    path: Path,
    media_type: str,
    *,
    required_for: list[str],
    canonical: bool,
) -> dict[str, Any]:
    return {
        "role": role,
        "path": path.name,
        "media_type": media_type,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "required_for": required_for,
        "canonical": canonical,
    }


def _artifact_descriptor(value: Mapping[str, Any]) -> ArtifactDescriptor:
    contract = value.get("array_contract")
    return ArtifactDescriptor(
        role=str(value["role"]),
        path=str(value["path"]),
        media_type=str(value["media_type"]),
        sha256=str(value["sha256"]),
        bytes=int(value["bytes"]),
        required_for=tuple(str(item) for item in value["required_for"]),
        canonical=bool(value["canonical"]),
        array_contract=contract if isinstance(contract, Mapping) else None,
    )


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _check_stored_size(path: Path, *, limit: int, limit_name: str, role: str) -> None:
    observed = path.stat().st_size
    if observed > limit:
        raise ArtifactResourceLimitError(
            operation="save result",
            limit=limit_name,
            configured=limit,
            observed=observed,
            artifact_role=role,
            path=path.name,
            action="Pass explicit trusted LoadLimits only when this output size is intended.",
        )


def _reject_existing(path: Path) -> None:
    if os.path.lexists(path):
        raise _collision(path, "already exists")


def _collision(path: Path, reason: str) -> OutputCollisionError:
    return OutputCollisionError(
        operation="save result",
        path=path,
        reason=reason,
        expected="a destination path that does not exist",
        actual="occupied output path",
        action="Choose another output directory; VAMOS never overwrites or merges run artifacts.",
    )


def _remove_owned_staging(staging: Path, *, parent: Path, token: str) -> None:
    if not os.path.lexists(staging):
        return
    expected_name = f".vamos-staging-{token}"
    try:
        resolved_parent = parent.resolve(strict=True)
        resolved_staging_parent = staging.parent.resolve(strict=True)
    except OSError:
        return
    if resolved_staging_parent != resolved_parent or not staging.name.endswith(expected_name):
        return
    shutil.rmtree(staging)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = ["store_succeeded_run"]
