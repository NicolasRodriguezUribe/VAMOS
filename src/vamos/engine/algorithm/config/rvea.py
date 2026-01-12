"""RVEA configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from .base import _SerializableConfig, _require_fields


class RVEAConfigDict(TypedDict):
    pop_size: int
    n_partitions: int
    alpha: float
    adapt_freq: float | None
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
class RVEAConfigData(_SerializableConfig["RVEAConfigDict"]):
    pop_size: int
    n_partitions: int
    alpha: float
    adapt_freq: float | None
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


class RVEAConfig:
    """
    Declarative configuration holder for RVEA settings.

    Examples:
        cfg = RVEAConfig.default()
        cfg = RVEAConfig().pop_size(100).n_partitions(12).fixed()
    """

    def __init__(self) -> None:
        self._cfg: dict[str, Any] = {}

    @classmethod
    def default(
        cls,
        pop_size: int = 100,
        n_var: int | None = None,
    ) -> "RVEAConfigData":
        """Create a default RVEA configuration."""
        mut_prob = 1.0 / n_var if n_var else 0.1
        return (
            cls()
            .pop_size(pop_size)
            .n_partitions(12)
            .alpha(2.0)
            .adapt_freq(0.1)
            .crossover("sbx", prob=1.0, eta=30.0)
            .mutation("pm", prob=mut_prob, eta=20.0)
            .fixed()
        )

    def pop_size(self, value: int) -> "RVEAConfig":
        self._cfg["pop_size"] = value
        return self

    def n_partitions(self, value: int) -> "RVEAConfig":
        self._cfg["n_partitions"] = value
        return self

    def alpha(self, value: float) -> "RVEAConfig":
        self._cfg["alpha"] = float(value)
        return self

    def adapt_freq(self, value: float | None) -> "RVEAConfig":
        self._cfg["adapt_freq"] = None if value is None else float(value)
        return self

    def crossover(self, method: str, **kwargs: Any) -> "RVEAConfig":
        self._cfg["crossover"] = (method, kwargs)
        return self

    def mutation(self, method: str, **kwargs: Any) -> "RVEAConfig":
        self._cfg["mutation"] = (method, kwargs)
        return self

    def repair(self, method: str, **kwargs: Any) -> "RVEAConfig":
        self._cfg["repair"] = (method, kwargs)
        return self

    def initializer(self, method: str, **kwargs: Any) -> "RVEAConfig":
        self._cfg["initializer"] = {"type": method, **kwargs}
        return self

    def mutation_prob_factor(self, value: float) -> "RVEAConfig":
        self._cfg["mutation_prob_factor"] = float(value)
        return self

    def constraint_mode(self, value: str) -> "RVEAConfig":
        self._cfg["constraint_mode"] = value
        return self

    def track_genealogy(self, enabled: bool = True) -> "RVEAConfig":
        self._cfg["track_genealogy"] = bool(enabled)
        return self

    def result_mode(self, value: str) -> "RVEAConfig":
        self._cfg["result_mode"] = str(value)
        return self

    def archive(self, size: int, **kwargs: Any) -> "RVEAConfig":
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

    def archive_type(self, value: str) -> "RVEAConfig":
        """Set archive pruning strategy."""
        self._cfg["archive_type"] = str(value)
        return self

    def fixed(self) -> RVEAConfigData:
        _require_fields(
            self._cfg,
            ("pop_size", "n_partitions", "alpha", "crossover", "mutation"),
            "RVEA",
        )
        return RVEAConfigData(
            pop_size=self._cfg["pop_size"],
            n_partitions=self._cfg.get("n_partitions", 12),
            alpha=float(self._cfg.get("alpha", 2.0)),
            adapt_freq=self._cfg.get("adapt_freq", 0.1),
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
