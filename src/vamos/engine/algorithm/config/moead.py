"""MOEA/D configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .base import _SerializableConfig, _require_fields


@dataclass(frozen=True)
class MOEADConfigData(_SerializableConfig):
    pop_size: int
    neighbor_size: int
    delta: float
    replace_limit: int
    crossover: Tuple[str, Dict[str, Any]]
    mutation: Tuple[str, Dict[str, Any]]
    aggregation: Tuple[str, Dict[str, Any]]
    weight_vectors: Dict[str, Optional[int | str]] | None
    engine: str
    constraint_mode: str = "feasibility"
    repair: Optional[Tuple[str, Dict[str, Any]]] = None
    initializer: Optional[Dict[str, Any]] = None
    mutation_prob_factor: Optional[float] = None
    track_genealogy: bool = False


class MOEADConfig:
    """
    Declarative configuration holder for MOEA/D settings.

    Examples:
        # Fluent builder
        cfg = MOEADConfig().pop_size(100).neighbor_size(20).fixed()

        # Quick default configuration
        cfg = MOEADConfig.default()

        # From dictionary
        cfg = MOEADConfig.from_dict({"pop_size": 100, "neighbor_size": 20})
    """

    def __init__(self) -> None:
        self._cfg: Dict[str, Any] = {}

    @classmethod
    def default(
        cls,
        pop_size: int = 100,
        n_var: int | None = None,
        engine: str = "numpy",
    ) -> "MOEADConfigData":
        """Create a default MOEA/D configuration with sensible defaults."""
        mut_prob = 1.0 / n_var if n_var else 0.1
        return (
            cls()
            .pop_size(pop_size)
            .neighbor_size(20)
            .delta(0.9)
            .replace_limit(2)
            .crossover("sbx", prob=0.9, eta=20.0)
            .mutation("pm", prob=mut_prob, eta=20.0)
            .aggregation("tchebycheff")
            .engine(engine)
            .fixed()
        )

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "MOEADConfigData":
        """Create configuration from a dictionary."""
        builder = cls()

        if "pop_size" in config:
            builder.pop_size(config["pop_size"])
        if "neighbor_size" in config:
            builder.neighbor_size(config["neighbor_size"])
        if "delta" in config:
            builder.delta(config["delta"])
        if "replace_limit" in config:
            builder.replace_limit(config["replace_limit"])

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

        # Handle aggregation
        if "aggregation" in config:
            agg = config["aggregation"]
            if isinstance(agg, tuple):
                builder.aggregation(agg[0], **agg[1])
            elif isinstance(agg, dict):
                method = agg.pop("method", agg.pop("type", "tchebycheff"))
                builder.aggregation(method, **agg)
            else:
                builder.aggregation(agg)

        if "engine" in config:
            builder.engine(config["engine"])
        if "constraint_mode" in config:
            builder.constraint_mode(config["constraint_mode"])

        return builder.fixed()

    def pop_size(self, value: int) -> "MOEADConfig":
        self._cfg["pop_size"] = value
        return self

    def neighbor_size(self, value: int) -> "MOEADConfig":
        self._cfg["neighbor_size"] = value
        return self

    def delta(self, value: float) -> "MOEADConfig":
        self._cfg["delta"] = value
        return self

    def replace_limit(self, value: int) -> "MOEADConfig":
        self._cfg["replace_limit"] = value
        return self

    def crossover(self, method: str, **kwargs) -> "MOEADConfig":
        self._cfg["crossover"] = (method, kwargs)
        return self

    def mutation(self, method: str, **kwargs) -> "MOEADConfig":
        self._cfg["mutation"] = (method, kwargs)
        return self

    def aggregation(self, method: str, **kwargs) -> "MOEADConfig":
        self._cfg["aggregation"] = (method, kwargs)
        return self

    def weight_vectors(
        self, *, path: Optional[str] = None, divisions: Optional[int] = None
    ) -> "MOEADConfig":
        self._cfg["weight_vectors"] = {"path": path, "divisions": divisions}
        return self

    def engine(self, value: str) -> "MOEADConfig":
        self._cfg["engine"] = value
        return self

    def constraint_mode(self, value: str) -> "MOEADConfig":
        self._cfg["constraint_mode"] = value
        return self

    def repair(self, method: str, **kwargs) -> "MOEADConfig":
        self._cfg["repair"] = (method, kwargs)
        return self

    def initializer(self, method: str, **kwargs) -> "MOEADConfig":
        self._cfg["initializer"] = {"type": method, **kwargs}
        return self

    def mutation_prob_factor(self, value: float) -> "MOEADConfig":
        self._cfg["mutation_prob_factor"] = float(value)
        return self

    def track_genealogy(self, enabled: bool = True) -> "MOEADConfig":
        self._cfg["track_genealogy"] = bool(enabled)
        return self

    def fixed(self) -> MOEADConfigData:
        _require_fields(
            self._cfg,
            (
                "pop_size",
                "neighbor_size",
                "delta",
                "replace_limit",
                "crossover",
                "mutation",
                "aggregation",
                "engine",
            ),
            "MOEA/D",
        )
        return MOEADConfigData(
            pop_size=self._cfg["pop_size"],
            neighbor_size=self._cfg["neighbor_size"],
            delta=self._cfg["delta"],
            replace_limit=self._cfg["replace_limit"],
            crossover=self._cfg["crossover"],
            mutation=self._cfg["mutation"],
            aggregation=self._cfg["aggregation"],
            weight_vectors=self._cfg.get("weight_vectors"),
            engine=self._cfg["engine"],
            constraint_mode=self._cfg.get("constraint_mode", "feasibility"),
            repair=self._cfg.get("repair"),
            initializer=self._cfg.get("initializer"),
            mutation_prob_factor=self._cfg.get("mutation_prob_factor"),
            track_genealogy=bool(self._cfg.get("track_genealogy", False)),
        )
