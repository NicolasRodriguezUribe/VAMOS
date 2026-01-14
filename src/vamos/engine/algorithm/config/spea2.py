"""SPEA2 configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import _SerializableConfig, _require_fields


@dataclass(frozen=True)
class SPEA2Config(_SerializableConfig):
    pop_size: int
    archive_size: int  # Internal archive (part of SPEA2 algorithm)
    crossover: tuple[str, dict[str, Any]]
    mutation: tuple[str, dict[str, Any]]
    selection: tuple[str, dict[str, Any]]
    k_neighbors: int | None = None
    repair: tuple[str, dict[str, Any]] | None = None
    initializer: dict[str, Any] | None = None
    mutation_prob_factor: float | None = None
    constraint_mode: str = "feasibility"
    track_genealogy: bool = False
    result_mode: str | None = None
    archive: dict[str, Any] | None = None  # External archive for results
    archive_type: str | None = None

    @classmethod
    def default(
        cls,
        pop_size: int = 100,
        n_var: int | None = None,
    ) -> SPEA2Config:
        """Create a default SPEA2 configuration."""
        mut_prob = 1.0 / n_var if n_var else 0.1
        return (
            cls.builder()
            .pop_size(pop_size)
            .archive_size(pop_size)
            .crossover("sbx", prob=1.0, eta=20.0)
            .mutation("pm", prob=mut_prob, eta=20.0)
            .selection("tournament")
            .build()
        )

    @classmethod
    def builder(cls) -> _SPEA2ConfigBuilder:
        return _SPEA2ConfigBuilder()


class _SPEA2ConfigBuilder:
    """
    Declarative configuration holder for SPEA2 settings.

    Examples:
        cfg = SPEA2Config.default()
        cfg = SPEA2Config.builder().pop_size(100).archive_size(100).build()
    """

    def __init__(self) -> None:
        self._cfg: dict[str, Any] = {}

    def pop_size(self, value: int) -> _SPEA2ConfigBuilder:
        self._cfg["pop_size"] = value
        return self

    def archive_size(self, value: int) -> _SPEA2ConfigBuilder:
        self._cfg["archive_size"] = value
        return self

    def crossover(self, method: str, **kwargs: Any) -> _SPEA2ConfigBuilder:
        self._cfg["crossover"] = (method, kwargs)
        return self

    def mutation(self, method: str, **kwargs: Any) -> _SPEA2ConfigBuilder:
        self._cfg["mutation"] = (method, kwargs)
        return self

    def selection(self, method: str, **kwargs: Any) -> _SPEA2ConfigBuilder:
        self._cfg["selection"] = (method, kwargs)
        return self

    def k_neighbors(self, value: int) -> _SPEA2ConfigBuilder:
        self._cfg["k_neighbors"] = value
        return self

    def repair(self, method: str, **kwargs: Any) -> _SPEA2ConfigBuilder:
        self._cfg["repair"] = (method, kwargs)
        return self

    def initializer(self, method: str, **kwargs: Any) -> _SPEA2ConfigBuilder:
        self._cfg["initializer"] = {"type": method, **kwargs}
        return self

    def mutation_prob_factor(self, value: float) -> _SPEA2ConfigBuilder:
        self._cfg["mutation_prob_factor"] = float(value)
        return self

    def constraint_mode(self, value: str) -> _SPEA2ConfigBuilder:
        self._cfg["constraint_mode"] = value
        return self

    def track_genealogy(self, enabled: bool = True) -> _SPEA2ConfigBuilder:
        self._cfg["track_genealogy"] = bool(enabled)
        return self

    def result_mode(self, value: str) -> _SPEA2ConfigBuilder:
        self._cfg["result_mode"] = str(value)
        return self

    def archive(self, size: int, **kwargs: Any) -> _SPEA2ConfigBuilder:
        """
        Configure an external archive for result storage.

        Note: This is separate from archive_size which is the internal SPEA2 archive.

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

    def external_archive_type(self, value: str) -> _SPEA2ConfigBuilder:
        """Set external archive pruning strategy."""
        self._cfg["archive_type"] = str(value)
        return self

    def build(self) -> SPEA2Config:
        _require_fields(
            self._cfg,
            ("pop_size", "archive_size", "crossover", "mutation", "selection"),
            "SPEA2",
        )
        return SPEA2Config(
            pop_size=self._cfg["pop_size"],
            archive_size=self._cfg["archive_size"],
            crossover=self._cfg["crossover"],
            mutation=self._cfg["mutation"],
            selection=self._cfg["selection"],
            k_neighbors=self._cfg.get("k_neighbors"),
            repair=self._cfg.get("repair"),
            initializer=self._cfg.get("initializer"),
            mutation_prob_factor=self._cfg.get("mutation_prob_factor"),
            constraint_mode=self._cfg.get("constraint_mode", "feasibility"),
            track_genealogy=bool(self._cfg.get("track_genealogy", False)),
            result_mode=self._cfg.get("result_mode", "non_dominated"),
            archive=self._cfg.get("archive"),
            archive_type=self._cfg.get("archive_type"),
        )
