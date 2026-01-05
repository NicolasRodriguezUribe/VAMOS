"""IBEA configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .base import _SerializableConfig, _require_fields


@dataclass(frozen=True)
class IBEAConfigData(_SerializableConfig):
    pop_size: int
    crossover: Tuple[str, Dict[str, Any]]
    mutation: Tuple[str, Dict[str, Any]]
    selection: Tuple[str, Dict[str, Any]]
    indicator: str
    kappa: float
    engine: str
    repair: Optional[Tuple[str, Dict[str, Any]]] = None
    initializer: Optional[Dict[str, Any]] = None
    mutation_prob_factor: Optional[float] = None
    constraint_mode: str = "feasibility"
    track_genealogy: bool = False
    result_mode: Optional[str] = None
    archive: Optional[Dict[str, Any]] = None
    archive_type: Optional[str] = None


class IBEAConfig:
    """Declarative configuration holder for IBEA settings."""

    def __init__(self) -> None:
        self._cfg: Dict[str, Any] = {}

    def pop_size(self, value: int) -> "IBEAConfig":
        self._cfg["pop_size"] = value
        return self

    def crossover(self, method: str, **kwargs) -> "IBEAConfig":
        self._cfg["crossover"] = (method, kwargs)
        return self

    def mutation(self, method: str, **kwargs) -> "IBEAConfig":
        self._cfg["mutation"] = (method, kwargs)
        return self

    def selection(self, method: str, **kwargs) -> "IBEAConfig":
        self._cfg["selection"] = (method, kwargs)
        return self

    def indicator(self, name: str) -> "IBEAConfig":
        self._cfg["indicator"] = name
        return self

    def kappa(self, value: float) -> "IBEAConfig":
        self._cfg["kappa"] = value
        return self

    def engine(self, value: str) -> "IBEAConfig":
        self._cfg["engine"] = value
        return self

    def repair(self, method: str, **kwargs) -> "IBEAConfig":
        self._cfg["repair"] = (method, kwargs)
        return self

    def initializer(self, method: str, **kwargs) -> "IBEAConfig":
        self._cfg["initializer"] = {"type": method, **kwargs}
        return self

    def mutation_prob_factor(self, value: float) -> "IBEAConfig":
        self._cfg["mutation_prob_factor"] = float(value)
        return self

    def constraint_mode(self, value: str) -> "IBEAConfig":
        self._cfg["constraint_mode"] = value
        return self

    def track_genealogy(self, enabled: bool = True) -> "IBEAConfig":
        self._cfg["track_genealogy"] = bool(enabled)
        return self

    def result_mode(self, value: str) -> "IBEAConfig":
        self._cfg["result_mode"] = str(value)
        return self

    def archive(self, size: int, **kwargs) -> "IBEAConfig":
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

    def archive_type(self, value: str) -> "IBEAConfig":
        """Set archive pruning strategy."""
        self._cfg["archive_type"] = str(value)
        return self

    def fixed(self) -> IBEAConfigData:
        _require_fields(
            self._cfg,
            ("pop_size", "crossover", "mutation", "selection", "indicator", "kappa", "engine"),
            "IBEA",
        )
        return IBEAConfigData(
            pop_size=self._cfg["pop_size"],
            crossover=self._cfg["crossover"],
            mutation=self._cfg["mutation"],
            selection=self._cfg["selection"],
            indicator=str(self._cfg["indicator"]),
            kappa=float(self._cfg["kappa"]),
            engine=self._cfg["engine"],
            repair=self._cfg.get("repair"),
            initializer=self._cfg.get("initializer"),
            mutation_prob_factor=self._cfg.get("mutation_prob_factor"),
            constraint_mode=self._cfg.get("constraint_mode", "feasibility"),
            track_genealogy=bool(self._cfg.get("track_genealogy", False)),
            result_mode=self._cfg.get("result_mode", "non_dominated"),
            archive=self._cfg.get("archive"),
            archive_type=self._cfg.get("archive_type"),
        )
