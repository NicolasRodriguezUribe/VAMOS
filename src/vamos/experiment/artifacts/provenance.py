"""Privacy-conscious provenance capture for v1 Python saves."""

from __future__ import annotations

import hashlib
import json
import locale
import os
import platform
import subprocess
import time
from collections.abc import Mapping
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np

from vamos.foundation.version import get_version

from .jsonio import normalize_json

_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def capture_environment(*, backend: str) -> dict[str, Any]:
    """Capture the contract-required, allowlisted environment description."""
    packages: dict[str, str] = {}
    for distribution in importlib_metadata.distributions():
        name = distribution.metadata["Name"]
        if isinstance(name, str) and name:
            packages[name.lower()] = distribution.version
    locale_name = locale.getlocale()[0] or "unknown"
    return {
        "document_type": "vamos.environment",
        "schema_version": "1.0.0",
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "operating_system": platform.system(),
            "release": platform.release(),
            "architecture": platform.machine(),
        },
        "packages": dict(sorted(packages.items())),
        "backend": {
            "name": backend,
            "package": _backend_distribution(backend),
        },
        "blas": _blas_metadata(),
        "threads": {key: os.environ[key] for key in _THREAD_VARIABLES if key in os.environ},
        "locale": locale_name,
        "timezone": time.tzname[0] if time.tzname else "unknown",
    }


def capture_provenance(
    *,
    backend: str,
    timestamps: Mapping[str, Any],
    entry_point: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(provenance, environment)`` without storing personal paths."""
    environment = capture_environment(backend=backend)
    distribution_sha, distribution_reason = _distribution_hash()
    distribution: dict[str, Any] = {
        "name": "vamos-optimization",
        "version": get_version(),
        "sha256": distribution_sha,
    }
    if distribution_reason is not None:
        distribution["hash_unavailable_reason"] = distribution_reason
    source = _source_metadata()
    provenance = {
        "implementation": {
            "vamos_version": get_version(),
            "distribution": distribution,
        },
        "source": source,
        "entry_point": normalize_json(
            entry_point
            or {
                "kind": "python_api",
                "python": {
                    "callable": "vamos.optimize",
                    "arguments_source": "requested_spec",
                },
            },
            field="$.provenance.entry_point",
        ),
        "environment_ref": "environment",
        "timestamps": normalize_json(timestamps, field="$.provenance.timestamps"),
    }
    return provenance, environment


def replayability_from_provenance(
    provenance: Mapping[str, Any],
    *,
    deterministic: bool,
) -> dict[str, Any]:
    source = provenance.get("source")
    implementation = provenance.get("implementation")
    reasons: list[dict[str, str]] = []
    level = "exact"
    dirty: object = "unknown"
    if isinstance(source, Mapping):
        dirty = source.get("dirty", "unknown")
    if dirty is True:
        level = "compatible"
        reasons.append({"code": "dirty_source_not_captured", "message": "Dirty source is recorded without executable source content."})
    elif dirty == "unknown":
        level = "compatible"
        reasons.append({"code": "source_identity_unknown", "message": "The installed source identity could not be verified."})
    distribution_hash: object = None
    if isinstance(implementation, Mapping):
        distribution = implementation.get("distribution")
        if isinstance(distribution, Mapping):
            distribution_hash = distribution.get("sha256")
    git_sha = source.get("git_sha") if isinstance(source, Mapping) else None
    if not git_sha and not distribution_hash:
        level = "compatible"
        reasons.append(
            {
                "code": "implementation_fingerprint_unavailable",
                "message": "Neither a Git revision nor an installed-distribution hash was available.",
            }
        )
    if not deterministic:
        level = "best_effort"
        reasons.append({"code": "execution_not_declared_deterministic", "message": "The execution path is not declared deterministic."})
    return {
        "declared_level": level,
        "deterministic": deterministic,
        "exact_requirements": [
            "same_resolved_spec",
            "same_implementation",
            "same_backend",
            "materially_equivalent_environment",
        ]
        if level == "exact"
        else [],
        "reasons": reasons,
    }


def _source_metadata() -> dict[str, Any]:
    root = _find_git_root(Path(__file__).resolve())
    if root is None:
        return {
            "git_sha": None,
            "dirty": "unknown",
            "diff_sha256": None,
            "tree_hash": None,
            "reason": "git_checkout_unavailable",
        }
    revision = _git_output(root, ("rev-parse", "HEAD"), text=True)
    status = _git_output(root, ("status", "--porcelain=v1", "-z"), text=False)
    if revision is None or status is None:
        return {
            "git_sha": revision,
            "dirty": "unknown",
            "diff_sha256": None,
            "tree_hash": None,
            "reason": "git_provenance_command_failed",
        }
    status_bytes = status if isinstance(status, bytes) else status.encode("utf-8")
    dirty = bool(status_bytes)
    diff_hash: str | None = None
    if dirty:
        diff = _git_output(root, ("diff", "--binary", "HEAD"), text=False)
        diff_bytes = diff if isinstance(diff, bytes) else (diff.encode("utf-8") if isinstance(diff, str) else b"")
        digest = hashlib.sha256()
        digest.update(diff_bytes)
        digest.update(b"\x00STATUS\x00")
        digest.update(status_bytes)
        diff_hash = digest.hexdigest()
    tree_inventory = _git_output(root, ("ls-tree", "-r", "--full-tree", "HEAD"), text=False)
    tree_bytes = tree_inventory if isinstance(tree_inventory, bytes) else b""
    tree_hash = hashlib.sha256(tree_bytes).hexdigest() if tree_bytes else None
    return {
        "git_sha": revision,
        "dirty": dirty,
        "diff_sha256": diff_hash,
        "tree_hash": tree_hash,
        "tree_hash_algorithm": "sha256-git-ls-tree" if tree_hash is not None else None,
    }


def _find_git_root(start: Path) -> Path | None:
    current = start.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _git_output(root: Path, args: tuple[str, ...], *, text: bool) -> str | bytes | None:
    try:
        completed = subprocess.run(
            ("git", *args),
            cwd=root,
            check=False,
            capture_output=True,
            text=text,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    if text:
        stdout = completed.stdout
        return stdout.strip() if isinstance(stdout, str) else None
    stdout_bytes = completed.stdout
    return bytes(stdout_bytes) if isinstance(stdout_bytes, (bytes, bytearray)) else None


def _distribution_hash() -> tuple[str | None, str | None]:
    try:
        distribution = importlib_metadata.distribution("vamos-optimization")
    except importlib_metadata.PackageNotFoundError:
        return None, "distribution_metadata_unavailable"
    direct_url = distribution.read_text("direct_url.json")
    if direct_url:
        try:
            direct_data = json.loads(direct_url)
        except json.JSONDecodeError:
            direct_data = {}
        directory_info = direct_data.get("dir_info") if isinstance(direct_data, dict) else None
        if isinstance(directory_info, dict) and directory_info.get("editable") is True:
            return None, "editable_install_uses_git_source_identity"
    files = distribution.files
    if not files:
        return None, "distribution_file_inventory_unavailable"
    digest = hashlib.sha256()
    matched = 0
    for relative in sorted(files, key=str):
        relative_name = relative.as_posix()
        if not relative_name.startswith("vamos/"):
            continue
        full_path = Path(str(distribution.locate_file(relative)))
        if not full_path.is_file():
            continue
        try:
            payload = full_path.read_bytes()
        except OSError:
            return None, "distribution_file_unreadable"
        digest.update(relative_name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(payload)
        matched += 1
    if matched == 0:
        return None, "distribution_package_files_unavailable"
    return digest.hexdigest(), None


def _backend_distribution(backend: str) -> dict[str, str | None]:
    package = {"numpy": "numpy", "numba": "numba", "moocore": "moocore"}.get(backend, backend)
    try:
        version = importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        version = None
    return {"name": package, "version": version}


def _blas_metadata() -> dict[str, Any]:
    config = getattr(np.__config__, "CONFIG", None)
    vendor = "unknown"
    if isinstance(config, Mapping):
        dependencies = config.get("Build Dependencies")
        if isinstance(dependencies, Mapping):
            blas = dependencies.get("blas")
            if isinstance(blas, Mapping):
                raw_vendor = blas.get("name") or blas.get("openblas configuration")
                if isinstance(raw_vendor, str) and raw_vendor:
                    vendor = raw_vendor
    integer_width: int | str = "unknown"
    if "ILP64" in vendor.upper() or "64" in vendor.upper():
        integer_width = 64
    return {"vendor": vendor, "integer_width": integer_width}


__all__ = ["capture_environment", "capture_provenance", "replayability_from_provenance"]
