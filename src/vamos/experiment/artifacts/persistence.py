"""Canonical public save/load operations for VAMOS v1 run artifacts."""

from __future__ import annotations

import math
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from vamos.experiment.optimization_result import OptimizationResult

from .bundle import snapshot_result_arrays
from .errors import IncompleteRunMetadataError
from .jsonio import canonical_json_bytes, normalize_json, sha256_bytes
from .manifest import DOCUMENT_TYPE, SCHEMA_VERSION
from .models import LoadLimits, StoredRun, VerifyMode
from .provenance import capture_provenance, replayability_from_provenance
from .reader import read_run
from .storage import store_succeeded_run


class ResultLike(Protocol):
    F: NDArray[Any] | None
    X: NDArray[Any] | None


def save_result(
    result: ResultLike,
    path: str | Path,
    *,
    requested_spec: Mapping[str, Any] | None = None,
    resolved_spec: Mapping[str, Any] | None = None,
    labels: Mapping[str, str] | None = None,
    limits: LoadLimits | None = None,
) -> StoredRun:
    """Persist a result as one immutable, relocatable v1 run directory."""
    active_limits = limits if limits is not None else LoadLimits()
    arrays = snapshot_result_arrays(result, limits=active_limits)
    recorded_requested, resolved, caller_supplied = _result_specs(
        result,
        requested_spec=requested_spec,
        resolved_spec=resolved_spec,
    )
    timestamps, runtime_ms = _timestamps(_result_meta_mapping(result))
    backend = _kernel_name(resolved)
    entry_point = _result_meta_mapping(result).get("run_artifact_entry_point")
    provenance, environment = capture_provenance(
        backend=backend,
        timestamps=timestamps,
        entry_point=entry_point if isinstance(entry_point, Mapping) else None,
    )
    deterministic = _declared_deterministic(resolved)
    replayability = replayability_from_provenance(provenance, deterministic=deterministic)
    unavailable_reason = _find_unavailable_reason(resolved)
    if unavailable_reason is not None:
        replayability = {
            "declared_level": "manual",
            "deterministic": deterministic,
            "exact_requirements": [],
            "reasons": [
                {
                    "code": "automatic_component_reconstruction_unavailable",
                    "message": unavailable_reason,
                }
            ],
        }
    if caller_supplied:
        provenance["entry_point"] = {
            "kind": "python_api",
            "python": {"callable": "vamos.save_result", "arguments_source": "caller_supplied_run_context"},
        }
        replayability = {
            "declared_level": "manual",
            "deterministic": deterministic,
            "exact_requirements": [],
            "reasons": [
                {
                    "code": "caller_supplied_execution_context",
                    "message": "The execution specification was supplied by the save_result caller.",
                }
            ],
        }
    task_id = "sha256:" + sha256_bytes(canonical_json_bytes(resolved))
    run_id = str(uuid.uuid4())
    manifest: dict[str, Any] = {
        "document_type": DOCUMENT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "task_id": task_id,
        "status": "succeeded",
        "timestamps": timestamps,
        "requested_spec": recorded_requested,
        "resolved_spec": resolved,
        "provenance": provenance,
        "replayability": replayability,
        "outcome": _outcome(result, arrays, resolved=resolved, runtime_ms=runtime_ms),
        "artifacts": [],
    }
    if labels is not None:
        manifest["labels"] = dict(labels)
    store_succeeded_run(
        Path(path),
        arrays=arrays,
        environment=environment,
        manifest_base=manifest,
        limits=active_limits,
    )
    return read_run(Path(path), verify="required", limits=active_limits)


def load_run(
    path: str | Path,
    *,
    verify: VerifyMode = "required",
    limits: LoadLimits | None = None,
) -> StoredRun:
    """Load immutable manifest access without resolving or executing code."""
    active_limits = limits if limits is not None else LoadLimits()
    return read_run(path, verify=verify, limits=active_limits)


def load_result(
    path: str | Path,
    *,
    verify: VerifyMode = "required",
    limits: LoadLimits | None = None,
) -> OptimizationResult:
    """Load the canonical numerical result; this never reruns optimization."""
    return load_run(path, verify=verify, limits=limits).result


def _result_specs(
    result: ResultLike,
    *,
    requested_spec: Mapping[str, Any] | None,
    resolved_spec: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    meta = _result_meta_mapping(result)
    captured_requested = meta.get("run_artifact_requested_spec")
    captured_resolved = meta.get("run_artifact_resolved_spec")
    supplied = requested_spec is not None or resolved_spec is not None
    if supplied and (requested_spec is None or resolved_spec is None):
        raise _incomplete_metadata(requested_spec, resolved_spec)
    requested = requested_spec if supplied else captured_requested
    resolved = resolved_spec if supplied else captured_resolved
    if isinstance(requested, Mapping) and isinstance(resolved, Mapping):
        normalized_requested = normalize_json(requested, field="$.requested_spec")
        normalized_resolved = normalize_json(resolved, field="$.resolved_spec")
        if not isinstance(normalized_requested, dict) or not isinstance(normalized_resolved, dict):
            raise AssertionError("normalized run specifications are not objects")
        return normalized_requested, normalized_resolved, supplied
    raise _incomplete_metadata(captured_requested, captured_resolved)


def _incomplete_metadata(requested: object, resolved: object) -> IncompleteRunMetadataError:
    missing = [field for field, value in (("requested_spec", requested), ("resolved_spec", resolved)) if not isinstance(value, Mapping)]
    return IncompleteRunMetadataError(
        operation="save result",
        field="$.run_context",
        reason="is incomplete",
        expected="both requested_spec and resolved_spec from the actual execution",
        actual={"missing": missing},
        action=(
            "Save an OptimizationResult returned by vamos.optimize(), or pass both requested_spec= and "
            "resolved_spec= with the complete execution context."
        ),
    )


def _timestamps(meta: Mapping[str, Any]) -> tuple[dict[str, str], float]:
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    raw = meta.get("run_artifact_timestamps")
    if not isinstance(raw, Mapping):
        return {"started_at": now, "completed_at": now}, 0.0
    started = raw.get("started_at")
    completed = raw.get("completed_at")
    runtime = raw.get("runtime_ms")
    runtime_ms = float(runtime) if isinstance(runtime, (int, float)) and not isinstance(runtime, bool) else 0.0
    if not math.isfinite(runtime_ms) or runtime_ms < 0:
        runtime_ms = 0.0
    return {
        "started_at": started if isinstance(started, str) else now,
        "completed_at": completed if isinstance(completed, str) else now,
    }, runtime_ms


def _outcome(
    result: ResultLike,
    arrays: Mapping[str, np.ndarray],
    *,
    resolved: Mapping[str, Any],
    runtime_ms: float,
) -> dict[str, Any]:
    data = _result_data_mapping(result)
    f_array = arrays["F"]
    x_array = arrays.get("X")
    evaluations = _integer(data.get("evaluations"))
    generations = _integer(data.get("generation"))
    checkpoint = data.get("checkpoint")
    if generations is None and isinstance(checkpoint, Mapping):
        generations = _integer(checkpoint.get("generation"))
    algorithm = resolved.get("algorithm")
    algorithm_config = algorithm.get("config") if isinstance(algorithm, Mapping) else None
    result_mode = algorithm_config.get("result_mode") if isinstance(algorithm_config, Mapping) else None
    termination = resolved.get("termination")
    termination_id = termination.get("component_id") if isinstance(termination, Mapping) else None
    termination_reason = "interrupted" if data.get("interrupted") is True else _component_suffix(termination_id)
    recorded_metrics = data.get("metrics")
    metrics: dict[str, Any] = (
        dict(normalize_json(recorded_metrics, field="$.outcome.metrics")) if isinstance(recorded_metrics, Mapping) else {}
    )
    for key in ("hv_reached", "best_hv", "hypervolume"):
        scalar = _scalar(data.get(key))
        if scalar is not None:
            metrics[key] = scalar
    outcome: dict[str, Any] = {
        "evaluations": evaluations,
        "generations": generations,
        "runtime_ms": runtime_ms,
        "termination_reason": termination_reason,
        "result_mode": result_mode or "unspecified",
        "interrupted": bool(data.get("interrupted", False)),
        "usable_result": True,
        "n_solutions": int(f_array.shape[0]),
        "n_objectives": int(f_array.shape[1]),
        "n_variables": int(x_array.shape[1]) if x_array is not None else None,
        "metrics": metrics,
    }
    return outcome


def _kernel_name(resolved: Mapping[str, Any]) -> str:
    backend = resolved.get("backend")
    kernel = backend.get("kernel") if isinstance(backend, Mapping) else None
    if isinstance(kernel, Mapping):
        resolution = kernel.get("resolution")
        if isinstance(resolution, Mapping) and isinstance(resolution.get("name"), str):
            return str(resolution["name"])
        suffix = _component_suffix(kernel.get("component_id"))
        if suffix is not None:
            return suffix
    return "unknown"


def _declared_deterministic(resolved: Mapping[str, Any]) -> bool:
    determinism = resolved.get("determinism")
    return isinstance(determinism, Mapping) and determinism.get("declared") is True


def _component_suffix(value: object) -> str | None:
    if not isinstance(value, str) or ":" not in value:
        return None
    return value.split(":", 1)[1].split("@", 1)[0]


def _integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return None


def _scalar(value: object) -> bool | int | float | str | None:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    return None


def _find_unavailable_reason(value: object) -> str | None:
    if isinstance(value, Mapping):
        reason = value.get("unavailable_reason")
        if isinstance(reason, str):
            return reason
        provider_type = value.get("type")
        if provider_type == "unavailable" and "distribution" in value:
            return "A required component provider is unavailable for automatic reconstruction."
        for item in value.values():
            found = _find_unavailable_reason(item)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = _find_unavailable_reason(item)
            if found is not None:
                return found
    return None


def _result_meta_mapping(result: ResultLike) -> Mapping[str, Any]:
    value = getattr(result, "meta", {})
    return value if isinstance(value, Mapping) else {}


def _result_data_mapping(result: ResultLike) -> Mapping[str, Any]:
    value = getattr(result, "data", {})
    return value if isinstance(value, Mapping) else {}


__all__ = ["ResultLike", "load_result", "load_run", "save_result"]
