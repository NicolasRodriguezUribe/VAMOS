"""SMPSO configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .base import _SerializableConfig, _require_fields


@dataclass(frozen=True)
class SMPSOConfigData(_SerializableConfig):
    pop_size: int
    archive_size: int  # Internal archive (part of SMPSO algorithm)
    mutation: Tuple[str, Dict[str, Any]]
    engine: str
    inertia: float = 0.5
    c1: float = 1.5
    c2: float = 1.5
    vmax_fraction: float = 0.5
    repair: Optional[Tuple[str, Dict[str, Any]]] = None
    initializer: Optional[Dict[str, Any]] = None
    constraint_mode: str = "feasibility"
    track_genealogy: bool = False
    result_mode: Optional[str] = None
    archive: Optional[Dict[str, Any]] = None  # External archive for results
    archive_type: Optional[str] = None


class SMPSOConfig:
    """Declarative configuration holder for SMPSO settings."""

    def __init__(self) -> None:
        self._cfg: Dict[str, Any] = {}

    def pop_size(self, value: int) -> "SMPSOConfig":
        self._cfg["pop_size"] = value
        return self

    def archive_size(self, value: int) -> "SMPSOConfig":
        self._cfg["archive_size"] = value
        return self

    def mutation(self, method: str, **kwargs) -> "SMPSOConfig":
        self._cfg["mutation"] = (method, kwargs)
        return self

    def engine(self, value: str) -> "SMPSOConfig":
        self._cfg["engine"] = value
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

    def repair(self, method: str, **kwargs) -> "SMPSOConfig":
        self._cfg["repair"] = (method, kwargs)
        return self

    def initializer(self, method: str, **kwargs) -> "SMPSOConfig":
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

    def archive(self, size: int, **kwargs) -> "SMPSOConfig":
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
            ("pop_size", "archive_size", "mutation", "engine"),
            "SMPSO",
        )
        return SMPSOConfigData(
            pop_size=self._cfg["pop_size"],
            archive_size=self._cfg["archive_size"],
            mutation=self._cfg["mutation"],
            engine=self._cfg["engine"],
            inertia=float(self._cfg.get("inertia", 0.5)),
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
