"""SMPSO configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from .base import _SerializableConfig, _require_fields


class SMPSOConfigDict(TypedDict):
    pop_size: int
    archive_size: int
    mutation: tuple[str, dict[str, Any]]
    inertia: float
    c1: float
    c2: float
    vmax_fraction: float
    repair: tuple[str, dict[str, Any]] | None
    initializer: dict[str, Any] | None
    constraint_mode: str
    track_genealogy: bool
    result_mode: str | None
    archive: dict[str, Any] | None
    archive_type: str | None


@dataclass(frozen=True)
class SMPSOConfigData(_SerializableConfig["SMPSOConfigDict"]):
    pop_size: int
    archive_size: int  # Internal archive (part of SMPSO algorithm)
    mutation: tuple[str, dict[str, Any]]
    inertia: float = 0.1
    c1: float = 1.5
    c2: float = 1.5
    vmax_fraction: float = 0.5
    repair: tuple[str, dict[str, Any]] | None = None
    initializer: dict[str, Any] | None = None
    constraint_mode: str = "feasibility"
    track_genealogy: bool = False
    result_mode: str | None = None
    archive: dict[str, Any] | None = None  # External archive for results
    archive_type: str | None = None


class SMPSOConfig:
    """Declarative configuration holder for SMPSO settings."""

    def __init__(self) -> None:
        self._cfg: dict[str, Any] = {}

    @classmethod
    def default(
        cls,
        pop_size: int = 100,
        n_var: int | None = None,
    ) -> "SMPSOConfigData":
        """Create a default SMPSO configuration."""
        mut_prob = 1.0 / n_var if n_var else 0.1
        return cls().pop_size(pop_size).archive_size(pop_size).mutation("pm", prob=mut_prob, eta=20.0).fixed()

    def pop_size(self, value: int) -> "SMPSOConfig":
        self._cfg["pop_size"] = value
        return self

    def archive_size(self, value: int) -> "SMPSOConfig":
        self._cfg["archive_size"] = value
        return self

    def mutation(self, method: str, **kwargs: Any) -> "SMPSOConfig":
        self._cfg["mutation"] = (method, kwargs)
        return self

    def inertia(self, value: float) -> "SMPSOConfig":
        self._cfg["inertia"] = value
        return self

    def c1(self, value: float) -> "SMPSOConfig":
        self._cfg["c1"] = value
        return self

    def c2(self, value: float) -> "SMPSOConfig":
        self._cfg["c2"] = value
        return self

    def vmax_fraction(self, value: float) -> "SMPSOConfig":
        self._cfg["vmax_fraction"] = value
        return self

    def repair(self, method: str, **kwargs: Any) -> "SMPSOConfig":
        self._cfg["repair"] = (method, kwargs)
        return self

    def initializer(self, method: str, **kwargs: Any) -> "SMPSOConfig":
        self._cfg["initializer"] = {"type": method, **kwargs}
        return self

    def constraint_mode(self, value: str) -> "SMPSOConfig":
        self._cfg["constraint_mode"] = value
        return self

    def track_genealogy(self, enabled: bool = True) -> "SMPSOConfig":
        self._cfg["track_genealogy"] = bool(enabled)
        return self

    def result_mode(self, value: str) -> "SMPSOConfig":
        self._cfg["result_mode"] = str(value)
        return self

    def archive(self, size: int, **kwargs: Any) -> "SMPSOConfig":
        """
        Configure an external archive for result storage.

        Note: This is separate from archive_size which is the internal SMPSO archive.

        Args:
            size: Archive size (required). <= 0 disables the archive.
            **kwargs: Optional configuration:
                - archive_type: "size_cap", "epsilon_grid", "hvc_prune", "hybrid"
                - prune_policy: "crowding", "hv_contrib", "random"
                - epsilon: Grid epsilon for epsilon_grid/hybrid types
        """
        if size <= 0:
            self._cfg["archive"] = {"size": 0}
            return self
        archive_cfg = {"size": int(size), **kwargs}
        self._cfg["archive"] = archive_cfg
        return self

    def external_archive_type(self, value: str) -> "SMPSOConfig":
        """Set external archive pruning strategy."""
        self._cfg["archive_type"] = str(value)
        return self

    def fixed(self) -> SMPSOConfigData:
        _require_fields(
            self._cfg,
            ("pop_size", "archive_size", "mutation"),
            "SMPSO",
        )
        return SMPSOConfigData(
            pop_size=self._cfg["pop_size"],
            archive_size=self._cfg["archive_size"],
            mutation=self._cfg["mutation"],
            inertia=float(self._cfg.get("inertia", 0.1)),
            c1=float(self._cfg.get("c1", 1.5)),
            c2=float(self._cfg.get("c2", 1.5)),
            vmax_fraction=float(self._cfg.get("vmax_fraction", 0.5)),
            repair=self._cfg.get("repair"),
            initializer=self._cfg.get("initializer"),
            constraint_mode=self._cfg.get("constraint_mode", "feasibility"),
            track_genealogy=bool(self._cfg.get("track_genealogy", False)),
            result_mode=self._cfg.get("result_mode", "non_dominated"),
            archive=self._cfg.get("archive"),
            archive_type=self._cfg.get("archive_type"),
        )
