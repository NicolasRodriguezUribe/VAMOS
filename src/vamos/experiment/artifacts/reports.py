"""Immutable public reports for verification and exact replay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .models import deep_thaw

CompatibilityLevel = Literal["exact", "compatible", "best_effort", "unavailable"]
ReplayabilityLevel = Literal["exact", "compatible", "best_effort", "manual", "unavailable"]
IntegrityStatus = Literal["valid", "invalid", "not_fully_checked"]
ComponentStatus = Literal["reconstructable", "manual", "unavailable"]


@dataclass(frozen=True, slots=True)
class CompatibilityFinding:
    """One material stored/current environment comparison."""

    field: str
    stored: Any
    current: Any
    classification: CompatibilityLevel
    explanation: str
    blocks_exact: bool
    action: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "stored": deep_thaw(self.stored),
            "current": deep_thaw(self.current),
            "classification": self.classification,
            "explanation": self.explanation,
            "blocks_exact": self.blocks_exact,
            "action": self.action,
        }


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    """Material compatibility of a stored run with the current runtime."""

    level: CompatibilityLevel
    findings: tuple[CompatibilityFinding, ...]

    @property
    def exact(self) -> bool:
        return self.level == "exact"

    def as_dict(self) -> dict[str, Any]:
        return {"level": self.level, "findings": [item.as_dict() for item in self.findings]}


@dataclass(frozen=True, slots=True)
class VerificationReason:
    """Structured reason contributing to effective replayability."""

    code: str
    field: str
    message: str
    action: str

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "field": self.field, "message": self.message, "action": self.action}


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """Independent verification dimensions for one canonical run."""

    root: Path
    run_id: str
    task_id: str
    status: str
    schema: str
    artifact_integrity: IntegrityStatus
    path_safety: IntegrityStatus
    numerical_bundle_safety: IntegrityStatus
    environment: CompatibilityReport
    component_reconstructability: ComponentStatus
    effective_replayability: ReplayabilityLevel
    reasons: tuple[VerificationReason, ...]
    optimization_executed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_type": "vamos.verification-report",
            "version": "1",
            "root": str(self.root),
            "run_id": self.run_id,
            "task_id": self.task_id,
            "status": self.status,
            "schema": self.schema,
            "artifact_integrity": self.artifact_integrity,
            "path_safety": self.path_safety,
            "numerical_bundle_safety": self.numerical_bundle_safety,
            "environment_compatibility": self.environment.as_dict(),
            "component_reconstructability": self.component_reconstructability,
            "effective_replayability": self.effective_replayability,
            "reasons": [reason.as_dict() for reason in self.reasons],
            "optimization_executed": self.optimization_executed,
        }


@dataclass(frozen=True, slots=True)
class ArrayComparison:
    """Bitwise comparison evidence for one canonical numerical role."""

    role: str
    exact: bool
    stored_dtype: str | None
    replay_dtype: str | None
    stored_shape: tuple[int, ...] | None
    replay_shape: tuple[int, ...] | None
    stored_sha256: str | None
    replay_sha256: str | None
    first_difference: tuple[int, ...] | None
    maximum_absolute_difference: float | None
    mismatch: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "exact": self.exact,
            "stored_dtype": self.stored_dtype,
            "replay_dtype": self.replay_dtype,
            "stored_shape": list(self.stored_shape) if self.stored_shape is not None else None,
            "replay_shape": list(self.replay_shape) if self.replay_shape is not None else None,
            "stored_sha256": self.stored_sha256,
            "replay_sha256": self.replay_sha256,
            "first_difference": list(self.first_difference) if self.first_difference is not None else None,
            "maximum_absolute_difference": self.maximum_absolute_difference,
            "mismatch": self.mismatch,
        }


@dataclass(frozen=True, slots=True)
class ReplayReport:
    """Outcome and evidence for a newly stored replay attempt."""

    source_root: Path
    output_root: Path
    source_run_id: str
    replay_run_id: str
    task_id: str
    source_manifest_sha256: str
    replay_plan_sha256: str
    exact: bool
    comparisons: tuple[ArrayComparison, ...]
    verification: VerificationReport
    optimization_executed: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_type": "vamos.replay-report",
            "version": "1",
            "source_root": str(self.source_root),
            "output_root": str(self.output_root),
            "source_run_id": self.source_run_id,
            "replay_run_id": self.replay_run_id,
            "task_id": self.task_id,
            "source_manifest_sha256": self.source_manifest_sha256,
            "replay_plan_sha256": self.replay_plan_sha256,
            "exact": self.exact,
            "comparisons": [item.as_dict() for item in self.comparisons],
            "verification": self.verification.as_dict(),
            "optimization_executed": self.optimization_executed,
        }


__all__ = [
    "ArrayComparison",
    "CompatibilityFinding",
    "CompatibilityLevel",
    "CompatibilityReport",
    "ComponentStatus",
    "IntegrityStatus",
    "ReplayReport",
    "ReplayabilityLevel",
    "VerificationReason",
    "VerificationReport",
]
