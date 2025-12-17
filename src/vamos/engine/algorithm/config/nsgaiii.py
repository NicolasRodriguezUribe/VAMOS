"""NSGA-III configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .base import _SerializableConfig, _require_fields


@dataclass(frozen=True)
class NSGAIIIConfigData(_SerializableConfig):
    pop_size: int
    crossover: Tuple[str, Dict[str, Any]]
    mutation: Tuple[str, Dict[str, Any]]
    selection: Tuple[str, Dict[str, Any]]
    reference_directions: Dict[str, Optional[int | str]]
    engine: str
    constraint_mode: str = "feasibility"
    repair: Optional[Tuple[str, Dict[str, Any]]] = None
    initializer: Optional[Dict[str, Any]] = None
    mutation_prob_factor: Optional[float] = None
    track_genealogy: bool = False


class NSGAIIIConfig:
    """
    Declarative configuration holder for NSGA-III settings.

    Examples:
        cfg = NSGAIIIConfig.default(n_obj=3)
        cfg = NSGAIIIConfig().pop_size(92).crossover("sbx", prob=0.9).fixed()
    """

    def __init__(self) -> None:
        self._cfg: Dict[str, Any] = {}

    @classmethod
    def default(
        cls,
        pop_size: int = 92,
        n_var: int | None = None,
        n_obj: int = 3,
        engine: str = "numpy",
    ) -> "NSGAIIIConfigData":
        """
        Create a default NSGA-III configuration.

        Args:
            pop_size: Population size (default: 92, matches 3-obj reference dirs)
            n_var: Number of variables (for mutation prob)
            n_obj: Number of objectives (for reference directions)
            engine: Backend engine
        """
        mut_prob = 1.0 / n_var if n_var else 0.1
        divisions = 12 if n_obj == 3 else 6
        return (
            cls()
            .pop_size(pop_size)
            .crossover("sbx", prob=0.9, eta=20.0)
            .mutation("pm", prob=mut_prob, eta=20.0)
            .selection("random")
            .reference_directions(divisions=divisions)
            .engine(engine)
            .fixed()
        )

    def pop_size(self, value: int) -> "NSGAIIIConfig":
        self._cfg["pop_size"] = value
        return self

    def crossover(self, method: str, **kwargs) -> "NSGAIIIConfig":
        self._cfg["crossover"] = (method, kwargs)
        return self

    def mutation(self, method: str, **kwargs) -> "NSGAIIIConfig":
        self._cfg["mutation"] = (method, kwargs)
        return self

    def selection(self, method: str, **kwargs) -> "NSGAIIIConfig":
        self._cfg["selection"] = (method, kwargs)
        return self

    def reference_directions(
        self,
        *,
        path: Optional[str] = None,
        divisions: Optional[int] = None,
    ) -> "NSGAIIIConfig":
        self._cfg["reference_directions"] = {"path": path, "divisions": divisions}
        return self

    def engine(self, value: str) -> "NSGAIIIConfig":
        self._cfg["engine"] = value
        return self

    def constraint_mode(self, value: str) -> "NSGAIIIConfig":
        self._cfg["constraint_mode"] = value
        return self

    def repair(self, method: str, **kwargs) -> "NSGAIIIConfig":
        self._cfg["repair"] = (method, kwargs)
        return self

    def initializer(self, method: str, **kwargs) -> "NSGAIIIConfig":
        self._cfg["initializer"] = {"type": method, **kwargs}
        return self

    def mutation_prob_factor(self, value: float) -> "NSGAIIIConfig":
        self._cfg["mutation_prob_factor"] = float(value)
        return self

    def track_genealogy(self, enabled: bool = True) -> "NSGAIIIConfig":
        self._cfg["track_genealogy"] = bool(enabled)
        return self

    def fixed(self) -> NSGAIIIConfigData:
        _require_fields(
            self._cfg,
            ("pop_size", "crossover", "mutation", "selection", "engine"),
            "NSGA-III",
        )
        ref_dirs = self._cfg.get("reference_directions", {})
        return NSGAIIIConfigData(
            pop_size=self._cfg["pop_size"],
            crossover=self._cfg["crossover"],
            mutation=self._cfg["mutation"],
            selection=self._cfg["selection"],
            reference_directions=ref_dirs,
            engine=self._cfg["engine"],
            constraint_mode=self._cfg.get("constraint_mode", "feasibility"),
            repair=self._cfg.get("repair"),
            initializer=self._cfg.get("initializer"),
            mutation_prob_factor=self._cfg.get("mutation_prob_factor"),
            track_genealogy=bool(self._cfg.get("track_genealogy", False)),
        )
