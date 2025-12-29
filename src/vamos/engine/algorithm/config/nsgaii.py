"""NSGA-II configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .base import _SerializableConfig, _require_fields


@dataclass(frozen=True)
class NSGAIIConfigData(_SerializableConfig):
    pop_size: int
    crossover: Tuple[str, Dict[str, Any]]
    mutation: Tuple[str, Dict[str, Any]]
    selection: Tuple[str, Dict[str, Any]]
    survival: str
    engine: str
    offspring_size: Optional[int] = None
    repair: Optional[Tuple[str, Dict[str, Any]]] = None
    archive: Optional[Dict[str, Any]] = None
    initializer: Optional[Dict[str, Any]] = None
    mutation_prob_factor: Optional[float] = None
    result_mode: Optional[str] = None
    archive_type: Optional[str] = None
    constraint_mode: str = "feasibility"
    track_genealogy: bool = False
    adaptive_operator_selection: Optional[Dict[str, Any]] = None


class NSGAIIConfig:
    """
    Declarative configuration holder for NSGA-II.
    Provides a fluent builder that yields an immutable NSGAIIConfigData.

    Examples:
        # Fluent builder
        cfg = NSGAIIConfig().pop_size(100).crossover("sbx", prob=0.9).fixed()

        # Quick default configuration
        cfg = NSGAIIConfig.default()

        # From dictionary
        cfg = NSGAIIConfig.from_dict({"pop_size": 100, "crossover": ("sbx", {"prob": 0.9})})
    """

    def __init__(self) -> None:
        self._cfg: Dict[str, Any] = {}

    @classmethod
    def default(
        cls,
        pop_size: int = 100,
        n_var: int | None = None,
        engine: str = "numpy",
    ) -> "NSGAIIConfigData":
        """
        Create a default NSGA-II configuration with sensible defaults.

        Args:
            pop_size: Population size (default: 100)
            n_var: Number of variables (used for mutation prob = 1/n_var)
            engine: Backend engine (default: "numpy")

        Returns:
            Frozen NSGAIIConfigData ready to use
        """
        mut_prob = 1.0 / n_var if n_var else 0.1
        return (
            cls()
            .pop_size(pop_size)
            .crossover("sbx", prob=0.9, eta=20.0)
            .mutation("pm", prob=mut_prob, eta=20.0)
            .selection("tournament")
            .survival("rank_crowding")
            .engine(engine)
            .fixed()
        )

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "NSGAIIConfigData":
        """
        Create configuration from a dictionary.

        Args:
            config: Dictionary with configuration keys:
                - pop_size (required): Population size
                - crossover: Tuple of (method, params) or dict with method and params
                - mutation: Tuple of (method, params) or dict with method and params
                - selection: Tuple of (method, params) or just method string
                - survival: Survival method string
                - engine: Backend engine string

        Returns:
            Frozen NSGAIIConfigData
        """
        builder = cls()

        if "pop_size" in config:
            builder.pop_size(config["pop_size"])

        # Handle crossover
        if "crossover" in config:
            cx = config["crossover"]
            if isinstance(cx, tuple):
                builder.crossover(cx[0], **cx[1])
            elif isinstance(cx, dict):
                method = cx.pop("method", cx.pop("type", "sbx"))
                builder.crossover(method, **cx)
            else:
                builder.crossover(cx)

        # Handle mutation
        if "mutation" in config:
            mut = config["mutation"]
            if isinstance(mut, tuple):
                builder.mutation(mut[0], **mut[1])
            elif isinstance(mut, dict):
                method = mut.pop("method", mut.pop("type", "pm"))
                builder.mutation(method, **mut)
            else:
                builder.mutation(mut)

        # Handle selection
        if "selection" in config:
            sel = config["selection"]
            if isinstance(sel, tuple):
                builder.selection(sel[0], **sel[1])
            elif isinstance(sel, dict):
                method = sel.pop("method", sel.pop("type", "tournament"))
                builder.selection(method, **sel)
            else:
                builder.selection(sel)

        # Simple string fields
        if "survival" in config:
            builder.survival(config["survival"])
        if "engine" in config:
            builder.engine(config["engine"])

        # Optional fields
        if "offspring_size" in config:
            builder.offspring_size(config["offspring_size"])
        if "repair" in config:
            rep = config["repair"]
            if isinstance(rep, tuple):
                builder.repair(rep[0], **rep[1])
            elif isinstance(rep, dict):
                method = rep.pop("method", rep.pop("type", "bounds"))
                builder.repair(method, **rep)
        if "archive" in config:
            builder.archive(config["archive"])
        if "archive_type" in config:
            builder.archive_type(config["archive_type"])
        if "constraint_mode" in config:
            builder.constraint_mode(config["constraint_mode"])
        if "track_genealogy" in config:
            builder.track_genealogy(config["track_genealogy"])
        if "adaptive_operator_selection" in config:
            builder.adaptive_operator_selection(config["adaptive_operator_selection"])

        return builder.fixed()

    def pop_size(self, value: int) -> "NSGAIIConfig":
        self._cfg["pop_size"] = value
        return self

    def crossover(self, method: str | tuple, params: dict | None = None, **kwargs) -> "NSGAIIConfig":
        if isinstance(method, tuple) and params is None and not kwargs:
            method, params = method
        cfg_kwargs = params or kwargs
        self._cfg["crossover"] = (method, cfg_kwargs)
        return self

    def mutation(self, method: str | tuple, params: dict | None = None, **kwargs) -> "NSGAIIConfig":
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

    def repair(self, method: str, **kwargs) -> "NSGAIIConfig":
        self._cfg["repair"] = (method, kwargs)
        return self

    def selection(self, method: str, **kwargs) -> "NSGAIIConfig":
        self._cfg["selection"] = (method, kwargs)
        return self

    def survival(self, method: str) -> "NSGAIIConfig":
        self._cfg["survival"] = method
        return self

    def engine(self, value: str) -> "NSGAIIConfig":
        self._cfg["engine"] = value
        return self

    def initializer(self, method: str, **kwargs) -> "NSGAIIConfig":
        self._cfg["initializer"] = {"type": method, **kwargs}
        return self

    def mutation_prob_factor(self, value: float) -> "NSGAIIConfig":
        self._cfg["mutation_prob_factor"] = float(value)
        return self

    def result_mode(self, value: str) -> "NSGAIIConfig":
        self._cfg["result_mode"] = str(value)
        return self

    def external_archive(self, *, size: int, archive_type: str = "hypervolume") -> "NSGAIIConfig":
        if size <= 0:
            raise ValueError("external archive size must be positive.")
        self._cfg["external_archive"] = {"size": int(size)}
        self._cfg["result_mode"] = "external_archive"
        self._cfg["archive_type"] = archive_type
        return self

    def archive_type(self, value: str) -> "NSGAIIConfig":
        """Set archive pruning strategy: 'hypervolume' or 'crowding'."""
        self._cfg["archive_type"] = str(value)
        return self

    def archive(self, size: int) -> "NSGAIIConfig":
        """
        Convenience alias to configure an external archive by size.
        A size <= 0 disables the archive.
        """
        if size <= 0:
            self._cfg["archive"] = {"size": 0}
            return self
        self._cfg["archive"] = {"size": int(size)}
        return self

    def constraint_mode(self, value: str) -> "NSGAIIConfig":
        """Set constraint handling mode: 'feasibility' or 'none'/'penalty'."""
        self._cfg["constraint_mode"] = value
        return self

    def track_genealogy(self, enabled: bool = True) -> "NSGAIIConfig":
        self._cfg["track_genealogy"] = bool(enabled)
        return self

    def adaptive_operator_selection(self, config: Dict[str, Any] | None) -> "NSGAIIConfig":
        if config is None:
            self._cfg["adaptive_operator_selection"] = None
        else:
            self._cfg["adaptive_operator_selection"] = dict(config)
        return self

    def fixed(self) -> NSGAIIConfigData:
        _require_fields(
            self._cfg,
            ("pop_size", "crossover", "mutation", "selection", "survival", "engine"),
            "NSGA-II",
        )
        archive_cfg = self._cfg.get("archive", self._cfg.get("external_archive"))
        return NSGAIIConfigData(
            pop_size=self._cfg["pop_size"],
            crossover=self._cfg["crossover"],
            mutation=self._cfg["mutation"],
            selection=self._cfg["selection"],
            survival=self._cfg["survival"],
            engine=self._cfg["engine"],
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
