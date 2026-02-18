"""IBEA configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vamos.archive import ExternalArchiveConfig
from .base import _SerializableConfig, _require_fields


@dataclass(frozen=True)
class IBEAConfig(_SerializableConfig):
    pop_size: int
    crossover: tuple[str, dict[str, Any]]
    mutation: tuple[str, dict[str, Any]]
    selection: tuple[str, dict[str, Any]]
    indicator: str
    kappa: float
    repair: tuple[str, dict[str, Any]] | None = None
    initializer: dict[str, Any] | None = None
    mutation_prob_factor: float | None = None
    constraint_mode: str = "feasibility"
    track_genealogy: bool = False
    result_mode: str | None = None
    external_archive: ExternalArchiveConfig | None = None

    @classmethod
    def default(
        cls,
        pop_size: int = 100,
        n_var: int | None = None,
    ) -> IBEAConfig:
        """Create a default IBEA configuration."""
        mut_prob = 1.0 / n_var if n_var else 0.1
        return (
            cls.builder()
            .pop_size(pop_size)
            .crossover("sbx", prob=1.0, eta=20.0)
            .mutation("pm", prob=mut_prob, eta=20.0)
            .selection("tournament")
            .indicator("eps")
            .kappa(1.0)
            .build()
        )

    @classmethod
    def builder(cls) -> _IBEAConfigBuilder:
        return _IBEAConfigBuilder()


class _IBEAConfigBuilder:
    """Declarative configuration holder for IBEA settings."""

    def __init__(self) -> None:
        self._cfg: dict[str, Any] = {}

    def pop_size(self, value: int) -> _IBEAConfigBuilder:
        self._cfg["pop_size"] = value
        return self

    def crossover(self, method: str, **kwargs: Any) -> _IBEAConfigBuilder:
        self._cfg["crossover"] = (method, kwargs)
        return self

    def mutation(self, method: str, **kwargs: Any) -> _IBEAConfigBuilder:
        self._cfg["mutation"] = (method, kwargs)
        return self

    def selection(self, method: str, **kwargs: Any) -> _IBEAConfigBuilder:
        self._cfg["selection"] = (method, kwargs)
        return self

    def indicator(self, name: str) -> _IBEAConfigBuilder:
        self._cfg["indicator"] = name
        return self

    def kappa(self, value: float) -> _IBEAConfigBuilder:
        self._cfg["kappa"] = value
        return self

    def repair(self, method: str, **kwargs: Any) -> _IBEAConfigBuilder:
        self._cfg["repair"] = (method, kwargs)
        return self

    def initializer(self, method: str, **kwargs: Any) -> _IBEAConfigBuilder:
        self._cfg["initializer"] = {"type": method, **kwargs}
        return self

    def mutation_prob_factor(self, value: float) -> _IBEAConfigBuilder:
        self._cfg["mutation_prob_factor"] = float(value)
        return self

    def constraint_mode(self, value: str) -> _IBEAConfigBuilder:
        self._cfg["constraint_mode"] = value
        return self

    def track_genealogy(self, enabled: bool = True) -> _IBEAConfigBuilder:
        self._cfg["track_genealogy"] = bool(enabled)
        return self

    def result_mode(self, value: str) -> _IBEAConfigBuilder:
        self._cfg["result_mode"] = str(value)
        return self

    def external_archive(self, capacity: int | None = None, **kwargs: Any) -> _IBEAConfigBuilder:
        """Configure an external archive.

        Args:
            capacity: Maximum number of solutions. ``None`` means unbounded.
            **kwargs: Forwarded to :class:`ExternalArchiveConfig`.
        """
        self._cfg["external_archive"] = ExternalArchiveConfig(capacity=capacity, **kwargs)
        return self

    def build(self) -> IBEAConfig:
        _require_fields(
            self._cfg,
            ("pop_size", "crossover", "mutation", "selection", "indicator", "kappa"),
            "IBEA",
        )
        return IBEAConfig(
            pop_size=self._cfg["pop_size"],
            crossover=self._cfg["crossover"],
            mutation=self._cfg["mutation"],
            selection=self._cfg["selection"],
            indicator=str(self._cfg["indicator"]),
            kappa=float(self._cfg["kappa"]),
            repair=self._cfg.get("repair"),
            initializer=self._cfg.get("initializer"),
            mutation_prob_factor=self._cfg.get("mutation_prob_factor"),
            constraint_mode=self._cfg.get("constraint_mode", "feasibility"),
            track_genealogy=bool(self._cfg.get("track_genealogy", False)),
            result_mode=self._cfg.get("result_mode", "non_dominated"),
            external_archive=self._cfg.get("external_archive"),
        )
