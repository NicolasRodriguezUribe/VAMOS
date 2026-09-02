"""Concise manifest-only run inspection."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .models import LoadLimits, deep_thaw
from .reader import read_run


def inspect_run(path: str | Path, *, limits: LoadLimits | None = None) -> dict[str, Any]:
    """Return a stable summary without materializing arrays or executing code."""
    active_limits = limits if limits is not None else LoadLimits()
    stored = read_run(path, verify="manifest", limits=active_limits)
    manifest = stored.manifest
    resolved = manifest.resolved_spec
    requested = manifest.requested_spec
    outcome = _mapping(manifest.get("outcome"))
    timestamps = _mapping(manifest.get("timestamps"))
    termination = _mapping(resolved.get("termination"))
    replayability = _mapping(manifest.get("replayability"))
    lineage = _mapping(manifest.get("lineage"))
    return {
        "document_type": "vamos.run-inspection",
        "version": "1",
        "root": str(stored.root),
        "schema_version": manifest.get("schema_version"),
        "status": manifest.status,
        "run_id": manifest.run_id,
        "task_id": manifest.task_id,
        "execution_kind": _execution_kind(manifest),
        "problem": _component_name(resolved.get("problem")),
        "algorithm": _component_name(resolved.get("algorithm")),
        "backend": _component_name(_mapping(resolved.get("backend")).get("kernel")),
        "requested_seed": _mapping(requested.get("defaults")).get("seed"),
        "resolved_seed": resolved.get("seed"),
        "population_size": _mapping(resolved.get("population")).get("initial_size"),
        "evaluation_budget": _budget(termination),
        "actual_evaluations": outcome.get("evaluations"),
        "termination": {
            "component": _component_name(termination),
            "reason": outcome.get("termination_reason"),
        },
        "timestamps": {
            "started_at": timestamps.get("started_at"),
            "completed_at": timestamps.get("completed_at"),
            "duration_ms": outcome.get("runtime_ms"),
        },
        "arrays": _arrays(manifest.artifacts),
        "replayability": replayability.get("declared_level"),
        "lineage": deep_thaw(lineage) if lineage else None,
        "full_artifact_verification": False,
        "optimization_executed": False,
        "recommended_next_command": f"vamos results verify {stored.root} --require-level exact",
    }


def _arrays(artifacts: tuple[Any, ...]) -> list[dict[str, Any]]:
    for descriptor in artifacts:
        if descriptor.role != "result_bundle" or not isinstance(descriptor.array_contract, Mapping):
            continue
        return [
            {
                "role": name,
                "shape": list(_mapping(spec).get("shape", [])),
                "dtype": _mapping(spec).get("dtype"),
            }
            for name, spec in sorted(descriptor.array_contract.items())
        ]
    return []


def _execution_kind(manifest: Mapping[str, Any]) -> str | None:
    lineage = _mapping(manifest.get("lineage"))
    if lineage:
        return str(lineage.get("execution_kind"))
    provenance = _mapping(manifest.get("provenance"))
    entry_point = _mapping(provenance.get("entry_point"))
    kind = entry_point.get("kind")
    return str(kind) if kind is not None else None


def _component_name(value: object) -> str | None:
    if not isinstance(value, Mapping):
        return None
    component_id = value.get("component_id")
    if not isinstance(component_id, str) or ":" not in component_id:
        return None
    return component_id.split(":", 1)[1].split("@", 1)[0]


def _budget(termination: Mapping[str, Any]) -> object:
    config = _mapping(termination.get("config"))
    return config.get("hard_max_evaluations", config.get("max_evaluations"))


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = ["inspect_run"]
