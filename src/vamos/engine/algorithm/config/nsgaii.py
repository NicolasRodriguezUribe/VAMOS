"""NSGA-II configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from .base import _SerializableConfig, _require_fields


class NSGAIIConfigDict(TypedDict):
    pop_size: int
    crossover: tuple[str, dict[str, Any]]
    mutation: tuple[str, dict[str, Any]]
    selection: tuple[str, dict[str, Any]]
    offspring_size: int | None
    repair: tuple[str, dict[str, Any]] | None
    archive: dict[str, Any] | None
    initializer: dict[str, Any] | None
    mutation_prob_factor: float | None
    result_mode: str | None
    archive_type: str | None
    constraint_mode: str
    track_genealogy: bool
    adaptive_operator_selection: dict[str, Any] | None


@dataclass(frozen=True)
class NSGAIIConfigData(_SerializableConfig["NSGAIIConfigDict"]):
    pop_size: int
    crossover: tuple[str, dict[str, Any]]
    mutation: tuple[str, dict[str, Any]]
    selection: tuple[str, dict[str, Any]]
    offspring_size: int | None = None
    repair: tuple[str, dict[str, Any]] | None = None
    archive: dict[str, Any] | None = None
    initializer: dict[str, Any] | None = None
    mutation_prob_factor: float | None = None
    result_mode: str | None = None
    archive_type: str | None = None
    constraint_mode: str = "feasibility"
    track_genealogy: bool = False
    adaptive_operator_selection: dict[str, Any] | None = None


class NSGAIIConfig:
    """
    Declarative configuration holder for NSGA-II.
    Provides a fluent builder that yields an immutable NSGAIIConfigData.

    Examples:
        # Fluent builder
        cfg = NSGAIIConfig().pop_size(100).crossover("sbx", prob=1.0).fixed()

        # Quick default configuration
        cfg = NSGAIIConfig.default()
    """

    def __init__(self) -> None:
        self._cfg: dict[str, Any] = {}

    @classmethod
    def default(
        cls,
        pop_size: int = 100,
        n_var: int | None = None,
    ) -> "NSGAIIConfigData":
        """
        Create a default NSGA-II configuration with sensible defaults.

        Args:
            pop_size: Population size (default: 100)
            n_var: Number of variables (used for mutation prob = 1/n_var)
        Returns:
            Frozen NSGAIIConfigData ready to use
        """
        mut_prob = 1.0 / n_var if n_var else 0.1
        return (
            cls()
            .pop_size(pop_size)
            .crossover("sbx", prob=1.0, eta=20.0)
            .mutation("pm", prob=mut_prob, eta=20.0)
            .selection("tournament")
            .fixed()
        )

    def pop_size(self, value: int) -> "NSGAIIConfig":
        self._cfg["pop_size"] = value
        return self

    def crossover(
        self,
        method: str | tuple[str, dict[str, Any]],
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "NSGAIIConfig":
        if isinstance(method, tuple) and params is None and not kwargs:
            method, params = method
        cfg_kwargs = params or kwargs
        self._cfg["crossover"] = (method, cfg_kwargs)
        return self

    def mutation(
        self,
        method: str | tuple[str, dict[str, Any]],
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "NSGAIIConfig":
        if isinstance(method, tuple) and params is None and not kwargs:
            method, params = method
        cfg_kwargs = params or kwargs
        self._cfg["mutation"] = (method, cfg_kwargs)
        return self

    def offspring_size(self, value: int) -> "NSGAIIConfig":
        if value <= 0:
            raise ValueError("offspring size must be positive.")
        self._cfg["offspring_size"] = value
        return self

    def repair(self, method: str, **kwargs: Any) -> "NSGAIIConfig":
        self._cfg["repair"] = (method, kwargs)
        return self

    def selection(self, method: str, **kwargs: Any) -> "NSGAIIConfig":
        self._cfg["selection"] = (method, kwargs)
        return self

    def initializer(self, method: str, **kwargs: Any) -> "NSGAIIConfig":
        self._cfg["initializer"] = {"type": method, **kwargs}
        return self

    def mutation_prob_factor(self, value: float) -> "NSGAIIConfig":
        self._cfg["mutation_prob_factor"] = float(value)
        return self

    def result_mode(self, value: str) -> "NSGAIIConfig":
        self._cfg["result_mode"] = str(value)
        return self

    def archive_type(self, value: str) -> "NSGAIIConfig":
        """Set archive pruning strategy: 'hypervolume' or 'crowding'."""
        self._cfg["archive_type"] = str(value)
        return self

    def archive(self, size: int, **kwargs: Any) -> "NSGAIIConfig":
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
        self._cfg.setdefault("result_mode", "external_archive")
        return self

    def constraint_mode(self, value: str) -> "NSGAIIConfig":
        """Set constraint handling mode: 'feasibility' or 'none'/'penalty'."""
        self._cfg["constraint_mode"] = value
        return self

    def track_genealogy(self, enabled: bool = True) -> "NSGAIIConfig":
        self._cfg["track_genealogy"] = bool(enabled)
        return self

    def adaptive_operator_selection(self, config: dict[str, Any] | None) -> "NSGAIIConfig":
        if config is None:
            self._cfg["adaptive_operator_selection"] = None
        else:
            self._cfg["adaptive_operator_selection"] = dict(config)
        return self

    def fixed(self) -> NSGAIIConfigData:
        _require_fields(
            self._cfg,
            ("pop_size", "crossover", "mutation", "selection"),
            "NSGA-II",
        )
        archive_cfg = self._cfg.get("archive")
        return NSGAIIConfigData(
            pop_size=self._cfg["pop_size"],
            crossover=self._cfg["crossover"],
            mutation=self._cfg["mutation"],
            selection=self._cfg["selection"],
            offspring_size=self._cfg.get("offspring_size"),
            repair=self._cfg.get("repair"),
            archive=archive_cfg,
            initializer=self._cfg.get("initializer"),
            mutation_prob_factor=self._cfg.get("mutation_prob_factor"),
            result_mode=self._cfg.get("result_mode", "non_dominated"),
            archive_type=self._cfg.get("archive_type"),
            constraint_mode=self._cfg.get("constraint_mode", "feasibility"),
            track_genealogy=bool(self._cfg.get("track_genealogy", False)),
            adaptive_operator_selection=self._cfg.get("adaptive_operator_selection"),
        )
