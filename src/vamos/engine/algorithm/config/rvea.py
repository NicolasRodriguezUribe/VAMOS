"""RVEA configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vamos.archive import ExternalArchiveConfig
from .base import _SerializableConfig, _require_fields


@dataclass(frozen=True)
class RVEAConfig(_SerializableConfig):
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
    external_archive: ExternalArchiveConfig | None = None

    @classmethod
    def default(
        cls,
        pop_size: int = 100,
        n_var: int | None = None,
    ) -> RVEAConfig:
        """Create a default RVEA configuration."""
        mut_prob = 1.0 / n_var if n_var else 0.1
        return (
            cls.builder()
            .pop_size(pop_size)
            .n_partitions(12)
            .alpha(2.0)
            .adapt_freq(0.1)
            .crossover("sbx", prob=1.0, eta=30.0)
            .mutation("pm", prob=mut_prob, eta=20.0)
            .build()
        )

    @classmethod
    def builder(cls) -> _RVEAConfigBuilder:
        return _RVEAConfigBuilder()


class _RVEAConfigBuilder:
    """
    Declarative configuration holder for RVEA settings.

    Examples:
        cfg = RVEAConfig.default()
        cfg = RVEAConfig.builder().pop_size(100).n_partitions(12).build()
    """

    def __init__(self) -> None:
        self._cfg: dict[str, Any] = {}

    def pop_size(self, value: int) -> _RVEAConfigBuilder:
        self._cfg["pop_size"] = value
        return self

    def n_partitions(self, value: int) -> _RVEAConfigBuilder:
        self._cfg["n_partitions"] = value
        return self

    def alpha(self, value: float) -> _RVEAConfigBuilder:
        self._cfg["alpha"] = float(value)
        return self

    def adapt_freq(self, value: float | None) -> _RVEAConfigBuilder:
        self._cfg["adapt_freq"] = None if value is None else float(value)
        return self

    def crossover(self, method: str, **kwargs: Any) -> _RVEAConfigBuilder:
        self._cfg["crossover"] = (method, kwargs)
        return self

    def mutation(self, method: str, **kwargs: Any) -> _RVEAConfigBuilder:
        self._cfg["mutation"] = (method, kwargs)
        return self

    def repair(self, method: str, **kwargs: Any) -> _RVEAConfigBuilder:
        self._cfg["repair"] = (method, kwargs)
        return self

    def initializer(self, method: str, **kwargs: Any) -> _RVEAConfigBuilder:
        self._cfg["initializer"] = {"type": method, **kwargs}
        return self

    def mutation_prob_factor(self, value: float) -> _RVEAConfigBuilder:
        self._cfg["mutation_prob_factor"] = float(value)
        return self

    def constraint_mode(self, value: str) -> _RVEAConfigBuilder:
        self._cfg["constraint_mode"] = value
        return self

    def track_genealogy(self, enabled: bool = True) -> _RVEAConfigBuilder:
        self._cfg["track_genealogy"] = bool(enabled)
        return self

    def result_mode(self, value: str) -> _RVEAConfigBuilder:
        mode = str(value).strip().lower()
        if mode not in {"non_dominated", "population"}:
            raise ValueError("result_mode must be 'non_dominated' or 'population'.")
        self._cfg["result_mode"] = mode
        return self

    def external_archive(self, capacity: int | None = None, **kwargs: Any) -> _RVEAConfigBuilder:
        """Configure an external archive.

        Args:
            capacity: Maximum number of solutions. ``None`` means unbounded.
            **kwargs: Forwarded to :class:`ExternalArchiveConfig`.
        """
        self._cfg["external_archive"] = ExternalArchiveConfig(capacity=capacity, **kwargs)
        return self

    def build(self) -> RVEAConfig:
        _require_fields(
            self._cfg,
            ("pop_size", "n_partitions", "alpha", "crossover", "mutation"),
            "RVEA",
        )
        return RVEAConfig(
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
            external_archive=self._cfg.get("external_archive"),
        )
