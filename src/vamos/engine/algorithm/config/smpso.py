"""SMPSO configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vamos.archive import ExternalArchiveConfig

from .base import ConstraintModeStr, ResultMode, _require_fields, _SerializableConfig


@dataclass(frozen=True)
class SMPSOConfig(_SerializableConfig):
    pop_size: int
    archive_size: int  # Internal archive (part of SMPSO algorithm)
    mutation: tuple[str, dict[str, Any]]
    inertia: float = 0.1
    c1: float = 1.5
    c2: float = 1.5
    vmax_fraction: float = 0.5
    repair: tuple[str, dict[str, Any]] | None = None
    initializer: dict[str, Any] | None = None
    constraint_mode: ConstraintModeStr = "feasibility"
    track_genealogy: bool = False
    result_mode: ResultMode | None = None
    external_archive: ExternalArchiveConfig | None = None

    @classmethod
    def default(
        cls,
        pop_size: int = 100,
        n_var: int | None = None,
    ) -> SMPSOConfig:
        """Create a default SMPSO configuration."""
        mut_prob = 1.0 / n_var if n_var else 0.1
        return cls.builder().pop_size(pop_size).archive_size(pop_size).mutation("pm", prob=mut_prob, eta=20.0).build()

    @classmethod
    def builder(cls) -> _SMPSOConfigBuilder:
        return _SMPSOConfigBuilder()


class _SMPSOConfigBuilder:
    """Declarative configuration holder for SMPSO settings."""

    def __init__(self) -> None:
        self._cfg: dict[str, Any] = {}

    def pop_size(self, value: int) -> _SMPSOConfigBuilder:
        self._cfg["pop_size"] = value
        return self

    def archive_size(self, value: int) -> _SMPSOConfigBuilder:
        self._cfg["archive_size"] = value
        return self

    def mutation(self, method: str, **kwargs: Any) -> _SMPSOConfigBuilder:
        self._cfg["mutation"] = (method, kwargs)
        return self

    def inertia(self, value: float) -> _SMPSOConfigBuilder:
        self._cfg["inertia"] = value
        return self

    def c1(self, value: float) -> _SMPSOConfigBuilder:
        self._cfg["c1"] = value
        return self

    def c2(self, value: float) -> _SMPSOConfigBuilder:
        self._cfg["c2"] = value
        return self

    def vmax_fraction(self, value: float) -> _SMPSOConfigBuilder:
        self._cfg["vmax_fraction"] = value
        return self

    def repair(self, method: str, **kwargs: Any) -> _SMPSOConfigBuilder:
        self._cfg["repair"] = (method, kwargs)
        return self

    def initializer(self, method: str, **kwargs: Any) -> _SMPSOConfigBuilder:
        self._cfg["initializer"] = {"type": method, **kwargs}
        return self

    def constraint_mode(self, value: str) -> _SMPSOConfigBuilder:
        self._cfg["constraint_mode"] = value
        return self

    def track_genealogy(self, enabled: bool = True) -> _SMPSOConfigBuilder:
        self._cfg["track_genealogy"] = bool(enabled)
        return self

    def result_mode(self, value: str) -> _SMPSOConfigBuilder:
        self._cfg["result_mode"] = str(value)
        return self

    def external_archive(self, capacity: int | None = None, **kwargs: Any) -> _SMPSOConfigBuilder:
        """Configure an external archive for result storage.

        Note: This is separate from ``archive_size`` which is the internal SMPSO archive.

        Args:
            capacity: Maximum number of solutions. ``None`` means unbounded.
            **kwargs: Forwarded to :class:`ExternalArchiveConfig`.
        """
        self._cfg["external_archive"] = ExternalArchiveConfig(capacity=capacity, **kwargs)
        return self

    def build(self) -> SMPSOConfig:
        _require_fields(
            self._cfg,
            ("pop_size", "archive_size", "mutation"),
            "SMPSO",
        )
        return SMPSOConfig(
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
            external_archive=self._cfg.get("external_archive"),
        )
