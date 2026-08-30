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
from .jsonio import canonical_json_bytes, normalize_json, sha256_bytes
from .manifest import DOCUMENT_TYPE, RESOLVED_SPEC_VERSION, SCHEMA_VERSION
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
    labels: Mapping[str, str] | None = None,
    limits: LoadLimits | None = None,
) -> StoredRun:
    """Persist a result as one immutable, relocatable v1 run directory."""
    active_limits = limits if limits is not None else LoadLimits()
    arrays = snapshot_result_arrays(result, limits=active_limits)
    recorded_requested, resolved, historical = _result_specs(result, arrays)
    if requested_spec is not None:
        normalized_requested = normalize_json(requested_spec, field="$.requested_spec")
        if not isinstance(normalized_requested, dict):
            raise AssertionError("normalize_json returned a non-object requested_spec")
        recorded_requested = normalized_requested
    timestamps, runtime_ms = _timestamps(_result_meta_mapping(result))
    backend = _kernel_name(resolved)
    provenance, environment = capture_provenance(backend=backend, timestamps=timestamps)
    deterministic = _declared_deterministic(resolved)
    replayability = replayability_from_provenance(provenance, deterministic=deterministic)
    unavailable_reason = _find_unavailable_reason(resolved)
    if unavailable_reason is not None and not historical:
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
    if historical:
        provenance["source"] = {
            "git_sha": None,
            "dirty": "unknown",
            "diff_sha256": None,
            "tree_hash": None,
            "reason": "original_execution_provenance_unavailable",
        }
        provenance["entry_point"] = {
            "kind": "python_api",
            "python": {"callable": "vamos.save_result", "arguments_source": "historical_resultlike"},
        }
        replayability = {
            "declared_level": "unavailable",
            "deterministic": False,
            "exact_requirements": [],
            "reasons": [
                {
                    "code": "original_execution_spec_unavailable",
                    "message": "The historical ResultLike did not carry a resolved execution specification.",
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


def _result_specs(result: ResultLike, arrays: Mapping[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any], bool]:
    meta = _result_meta_mapping(result)
    requested = meta.get("run_artifact_requested_spec")
    resolved = meta.get("run_artifact_resolved_spec")
    if isinstance(requested, Mapping) and isinstance(resolved, Mapping):
        normalized_requested = normalize_json(requested, field="$.requested_spec")
        normalized_resolved = normalize_json(resolved, field="$.resolved_spec")
        if not isinstance(normalized_requested, dict) or not isinstance(normalized_resolved, dict):
            raise AssertionError("normalized run specifications are not objects")
        return normalized_requested, normalized_resolved, False
    return _historical_specs(result, arrays)


def _historical_specs(
    result: ResultLike,
    arrays: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    f_array = arrays["F"]
    x_array = arrays.get("X")
    seed_value = _result_meta_mapping(result).get("seed")
    seed = int(seed_value) if isinstance(seed_value, int) and not isinstance(seed_value, bool) else 0
    unknown_resolution = {
        "contract_version": 1,
        "available": False,
        "reason": "historical_resultlike_metadata_unavailable",
    }

    def component(kind: str, name: str, config: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "kind": kind,
            "component_id": f"vamos.unavailable:{name}@1",
            "provider": {"type": "unavailable", "distribution": None},
            "config": dict(config),
            "resolution": dict(unknown_resolution),
        }

    requested = {
        "version": "1",
        "source": {
            "kind": "historical_resultlike",
            "configuration": "unavailable",
            "reason": "save_result caller supplied arrays without a captured execution spec",
        },
    }
    resolved = {
        "spec_version": RESOLVED_SPEC_VERSION,
        "problem": component(
            "problem",
            "problem",
            {
                "n_obj": int(f_array.shape[1]),
                "n_var": int(x_array.shape[1]) if x_array is not None else None,
                "constraint_convention": "g_lte_0",
            },
        ),
        "algorithm": component("algorithm", "algorithm", {}),
        "operators": {},
        "backend": {
            "kernel": component("kernel_backend", "kernel", {}),
            "evaluation": component("evaluation_backend", "evaluation", {}),
        },
        "termination": component("termination", "termination", {}),
        "seed": seed,
        "population": {
            "initial_size": None,
            "offspring_size": None,
            "archive_size": None,
        },
        "defaults_applied": [
            {
                "field": "/seed",
                "source": "unavailable",
                "reason": "structural sentinel only; the original seed is unknown"
                if seed_value is None
                else "recorded from result metadata",
            }
        ],
        "determinism": {
            "declared": False,
            "rng": "unavailable",
            "seed_evidence": "unavailable" if seed_value is None else "result_metadata",
        },
    }
    return requested, resolved, True


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
    metrics: dict[str, Any] = {}
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
