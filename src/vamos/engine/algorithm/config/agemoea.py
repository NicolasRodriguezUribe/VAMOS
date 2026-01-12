"""AGE-MOEA configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from .base import _SerializableConfig, _require_fields


class AGEMOEAConfigDict(TypedDict):
    pop_size: int
    crossover: tuple[str, dict[str, Any]]
    mutation: tuple[str, dict[str, Any]]
    repair: tuple[str, dict[str, Any]] | None
    initializer: dict[str, Any] | None
    mutation_prob_factor: float | None
    constraint_mode: str
    track_genealogy: bool
    result_mode: str | None
    archive: dict[str, Any] | None
    archive_type: str | None


@dataclass(frozen=True)
class AGEMOEAConfigData(_SerializableConfig["AGEMOEAConfigDict"]):
    pop_size: int
    crossover: tuple[str, dict[str, Any]]
    mutation: tuple[str, dict[str, Any]]
    repair: tuple[str, dict[str, Any]] | None = None
    initializer: dict[str, Any] | None = None
    mutation_prob_factor: float | None = None
    constraint_mode: str = "feasibility"
    track_genealogy: bool = False
    result_mode: str | None = None
    archive: dict[str, Any] | None = None
    archive_type: str | None = None


class AGEMOEAConfig:
    """
    Declarative configuration holder for AGE-MOEA settings.

    Examples:
        cfg = AGEMOEAConfig.default()
        cfg = AGEMOEAConfig().pop_size(100).fixed()
    """

    def __init__(self) -> None:
        self._cfg: dict[str, Any] = {}

    @classmethod
    def default(
        cls,
        pop_size: int = 100,
        n_var: int | None = None,
    ) -> "AGEMOEAConfigData":
        """Create a default AGE-MOEA configuration."""
        mut_prob = 1.0 / n_var if n_var else 0.1
        return cls().pop_size(pop_size).crossover("sbx", prob=0.9, eta=15.0).mutation("pm", prob=mut_prob, eta=20.0).fixed()

    def pop_size(self, value: int) -> "AGEMOEAConfig":
        self._cfg["pop_size"] = value
        return self

    def crossover(self, method: str, **kwargs: Any) -> "AGEMOEAConfig":
        self._cfg["crossover"] = (method, kwargs)
        return self

    def mutation(self, method: str, **kwargs: Any) -> "AGEMOEAConfig":
        self._cfg["mutation"] = (method, kwargs)
        return self

    def repair(self, method: str, **kwargs: Any) -> "AGEMOEAConfig":
        self._cfg["repair"] = (method, kwargs)
        return self

    def initializer(self, method: str, **kwargs: Any) -> "AGEMOEAConfig":
        self._cfg["initializer"] = {"type": method, **kwargs}
        return self

    def mutation_prob_factor(self, value: float) -> "AGEMOEAConfig":
        self._cfg["mutation_prob_factor"] = float(value)
        return self

    def constraint_mode(self, value: str) -> "AGEMOEAConfig":
        self._cfg["constraint_mode"] = value
        return self

    def track_genealogy(self, enabled: bool = True) -> "AGEMOEAConfig":
        self._cfg["track_genealogy"] = bool(enabled)
        return self

    def result_mode(self, value: str) -> "AGEMOEAConfig":
        self._cfg["result_mode"] = str(value)
        return self

    def archive(self, size: int, **kwargs: Any) -> "AGEMOEAConfig":
        """
        Configure an external archive.

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

    def archive_type(self, value: str) -> "AGEMOEAConfig":
        """Set archive pruning strategy."""
        self._cfg["archive_type"] = str(value)
        return self

    def fixed(self) -> AGEMOEAConfigData:
        _require_fields(
            self._cfg,
            ("pop_size", "crossover", "mutation"),
            "AGE-MOEA",
        )
        return AGEMOEAConfigData(
            pop_size=self._cfg["pop_size"],
            crossover=self._cfg["crossover"],
            mutation=self._cfg["mutation"],
            repair=self._cfg.get("repair"),
            initializer=self._cfg.get("initializer"),
            mutation_prob_factor=self._cfg.get("mutation_prob_factor"),
            constraint_mode=self._cfg.get("constraint_mode", "feasibility"),
            track_genealogy=bool(self._cfg.get("track_genealogy", False)),
            result_mode=self._cfg.get("result_mode", "non_dominated"),
            archive=self._cfg.get("archive"),
            archive_type=self._cfg.get("archive_type"),
        )
