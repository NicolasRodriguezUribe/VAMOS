"""Immutable data models for VAMOS v1 run artifacts."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, fields
from functools import cached_property
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from .errors import ManifestValidationError

if TYPE_CHECKING:
    from vamos.experiment.optimization_result import OptimizationResult

VerifyMode = Literal["manifest", "required", "all"]


@dataclass(frozen=True, slots=True)
class LoadLimits:
    """Finite defensive limits used by trusted v1 readers.

    Callers may pass a different instance explicitly when they trust a larger
    artifact. Normal loading never increases these limits automatically.
    """

    max_manifest_bytes: int = 8 * 1024 * 1024
    max_environment_bytes: int = 16 * 1024 * 1024
    max_artifact_bytes: int = 512 * 1024 * 1024
    max_artifacts: int = 128
    max_json_depth: int = 64
    max_zip_members: int = 128
    max_arrays: int = 64
    max_total_uncompressed_bytes: int = 1024 * 1024 * 1024
    max_array_bytes: int = 512 * 1024 * 1024
    max_total_elements: int = 100_000_000
    max_array_dimensions: int = 8
    max_npy_header_bytes: int = 64 * 1024
    max_compression_ratio: float = 1000.0

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name == "max_compression_ratio":
                valid = not isinstance(value, bool) and isinstance(value, (int, float)) and value > 0
            else:
                valid = not isinstance(value, bool) and isinstance(value, int) and value > 0
            if not valid:
                raise ValueError(f"LoadLimits.{item.name} must be a positive number.")


def deep_freeze(value: Any) -> Any:
    """Return an immutable defensive copy of a JSON-compatible value."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(deep_freeze(item) for item in value)
    return value


def deep_thaw(value: Any) -> Any:
    """Return a mutable JSON-compatible copy of a frozen value."""
    if isinstance(value, Mapping):
        return {str(key): deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [deep_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ArtifactDescriptor:
    """Validated immutable manifest reference to one stored artifact."""

    role: str
    path: str
    media_type: str
    sha256: str
    bytes: int
    required_for: tuple[str, ...]
    canonical: bool
    array_contract: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": self.role,
            "path": self.path,
            "media_type": self.media_type,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "required_for": list(self.required_for),
            "canonical": self.canonical,
        }
        if self.array_contract is not None:
            payload["array_contract"] = deep_thaw(self.array_contract)
        return payload


@dataclass(frozen=True, slots=True)
class ResolvedRunSpec(Mapping[str, Any]):
    """Immutable effective configuration used to derive a run task ID."""

    _data: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ResolvedRunSpec:
        return cls(deep_freeze(value))

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def as_dict(self) -> dict[str, Any]:
        thawed = deep_thaw(self._data)
        if not isinstance(thawed, dict):
            raise AssertionError("deep_thaw returned a non-object resolved spec")
        return thawed


@dataclass(frozen=True, slots=True)
class RunManifest(Mapping[str, Any]):
    """Immutable validated envelope for one v1 execution attempt."""

    _data: Mapping[str, Any]
    resolved_spec: ResolvedRunSpec
    artifacts: tuple[ArtifactDescriptor, ...]

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    @property
    def run_id(self) -> str:
        return str(self._data["run_id"])

    @property
    def task_id(self) -> str:
        return str(self._data["task_id"])

    @property
    def status(self) -> str:
        return str(self._data["status"])

    @property
    def requested_spec(self) -> Mapping[str, Any]:
        value = self._data["requested_spec"]
        if not isinstance(value, Mapping):
            raise ManifestValidationError(
                operation="access manifest",
                field="$.requested_spec",
                reason="is not an object",
                expected="JSON object",
                actual=type(value).__name__,
                action="Restore a valid v1 manifest.",
            )
        return value

    def artifact(self, role: str) -> ArtifactDescriptor | None:
        """Return the singleton descriptor for ``role`` when present."""
        for descriptor in self.artifacts:
            if descriptor.role == role:
                return descriptor
        return None

    def as_dict(self) -> dict[str, Any]:
        thawed = deep_thaw(self._data)
        if not isinstance(thawed, dict):
            raise AssertionError("deep_thaw returned a non-object manifest")
        return thawed


@dataclass(frozen=True)
class StoredRun:
    """Immutable data-only handle with lazy, side-effect-free artifact access."""

    root: Path
    manifest: RunManifest
    _result_loader: Callable[[], OptimizationResult] = field(repr=False, compare=False)
    _environment_loader: Callable[[], Mapping[str, Any]] = field(repr=False, compare=False)

    @property
    def status(self) -> str:
        return self.manifest.status

    @cached_property
    def result(self) -> OptimizationResult:
        """Load and cache the canonical result bundle without executing code."""
        return self._result_loader()

    @cached_property
    def environment(self) -> Mapping[str, Any]:
        """Load and cache the immutable captured environment document."""
        return self._environment_loader()


__all__ = [
    "ArtifactDescriptor",
    "LoadLimits",
    "ResolvedRunSpec",
    "RunManifest",
    "StoredRun",
    "VerifyMode",
]
