"""AGE-MOEA configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, overload

from vamos.engine.archive import ExternalArchiveConfig

from .base import ConstraintModeStr, ResultMode, _build_external_archive_config, _require_fields, _SerializableConfig, _validate_operators
from .types import CrossoverName, InitializerName, MutationName, RepairConfigValue, RepairName


class _AGEMOEAConfigBuilder:
    """
    Fluent builder for AGE-MOEA configs.
    """

    def __init__(self) -> None:
        self._cfg: dict[str, Any] = {}

    def pop_size(self, value: int) -> _AGEMOEAConfigBuilder:
        self._cfg["pop_size"] = value
        return self

    @overload
    def crossover(self, method: CrossoverName, **kwargs: Any) -> _AGEMOEAConfigBuilder: ...

    @overload
    def crossover(self, method: str, **kwargs: Any) -> _AGEMOEAConfigBuilder: ...

    def crossover(self, method: str, **kwargs: Any) -> _AGEMOEAConfigBuilder:
        self._cfg["crossover"] = (method, kwargs)
        return self

    @overload
    def mutation(self, method: MutationName, **kwargs: Any) -> _AGEMOEAConfigBuilder: ...

    @overload
    def mutation(self, method: str, **kwargs: Any) -> _AGEMOEAConfigBuilder: ...

    def mutation(self, method: str, **kwargs: Any) -> _AGEMOEAConfigBuilder:
        self._cfg["mutation"] = (method, kwargs)
        return self

    @overload
    def repair(self, method: RepairName, **kwargs: Any) -> _AGEMOEAConfigBuilder: ...

    @overload
    def repair(self, method: str, **kwargs: Any) -> _AGEMOEAConfigBuilder: ...

    def repair(self, method: str, **kwargs: Any) -> _AGEMOEAConfigBuilder:
        self._cfg["repair"] = (method, kwargs)
        return self

    @overload
    def initializer(self, method: InitializerName, **kwargs: Any) -> _AGEMOEAConfigBuilder: ...

    @overload
    def initializer(self, method: str, **kwargs: Any) -> _AGEMOEAConfigBuilder: ...

    def initializer(self, method: str, **kwargs: Any) -> _AGEMOEAConfigBuilder:
        self._cfg["initializer"] = {"type": method, **kwargs}
        return self

    def mutation_prob_factor(self, value: float) -> _AGEMOEAConfigBuilder:
        self._cfg["mutation_prob_factor"] = float(value)
        return self

    def constraint_mode(self, value: ConstraintModeStr) -> _AGEMOEAConfigBuilder:
        self._cfg["constraint_mode"] = value
        return self

    def track_genealogy(self, enabled: bool = True) -> _AGEMOEAConfigBuilder:
        self._cfg["track_genealogy"] = bool(enabled)
        return self

    def result_mode(self, value: ResultMode) -> _AGEMOEAConfigBuilder:
        mode = str(value).strip().lower()
        if mode not in {"non_dominated", "population"}:
            raise ValueError("result_mode must be 'non_dominated' or 'population'.")
        self._cfg["result_mode"] = mode
        return self

    def external_archive(self, capacity: int | None = None, **kwargs: Any) -> _AGEMOEAConfigBuilder:
        """Configure an external archive.

        Args:
            capacity: Maximum number of solutions. ``None`` means unbounded.
            **kwargs: Forwarded to :class:`ExternalArchiveConfig`.
        """
        self._cfg["external_archive"] = _build_external_archive_config(capacity, kwargs)
        self._cfg.setdefault("result_mode", "non_dominated")
        return self

    def build(self) -> AGEMOEAConfig:
        _require_fields(
            self._cfg,
            ("pop_size", "crossover", "mutation"),
            "AGE-MOEA",
        )
        _validate_operators(self._cfg)
        return AGEMOEAConfig(
            pop_size=self._cfg["pop_size"],
            crossover=self._cfg["crossover"],
            mutation=self._cfg["mutation"],
            repair=self._cfg.get("repair", "auto"),
            initializer=self._cfg.get("initializer"),
            mutation_prob_factor=self._cfg.get("mutation_prob_factor"),
            constraint_mode=self._cfg.get("constraint_mode", "feasibility"),
            track_genealogy=bool(self._cfg.get("track_genealogy", False)),
            result_mode=self._cfg.get("result_mode", "non_dominated"),
            external_archive=self._cfg.get("external_archive"),
        )


@dataclass(frozen=True)
class AGEMOEAConfig(_SerializableConfig):
    pop_size: int
    crossover: tuple[str, dict[str, Any]]
    mutation: tuple[str, dict[str, Any]]
    repair: RepairConfigValue = "auto"
    initializer: dict[str, Any] | None = None
    mutation_prob_factor: float | None = None
    constraint_mode: ConstraintModeStr = "feasibility"
    track_genealogy: bool = False
    result_mode: ResultMode | None = None
    external_archive: ExternalArchiveConfig | None = None

    @classmethod
    def default(
        cls,
        pop_size: int = 100,
        n_var: int | None = None,
    ) -> AGEMOEAConfig:
        """Create a default AGE-MOEA configuration."""
        mut_prob = 1.0 / n_var if n_var else 0.1
        return cls.builder().pop_size(pop_size).crossover("sbx", prob=0.9, eta=15.0).mutation("pm", prob=mut_prob, eta=20.0).build()

    @classmethod
    def builder(cls) -> _AGEMOEAConfigBuilder:
        return _AGEMOEAConfigBuilder()
