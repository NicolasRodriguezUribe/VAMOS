"""SMS-EMOA configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .base import _SerializableConfig, _require_fields


@dataclass(frozen=True)
class SMSEMOAConfigData(_SerializableConfig):
    pop_size: int
    crossover: Tuple[str, Dict[str, Any]]
    mutation: Tuple[str, Dict[str, Any]]
    selection: Tuple[str, Dict[str, Any]]
    reference_point: Dict[str, Any]
    engine: str
    constraint_mode: str = "feasibility"
    repair: Optional[Tuple[str, Dict[str, Any]]] = None
    initializer: Optional[Dict[str, Any]] = None
    mutation_prob_factor: Optional[float] = None
    track_genealogy: bool = False
    result_mode: Optional[str] = None
    archive_type: Optional[str] = None
    archive: Optional[Dict[str, Any]] = None


class SMSEMOAConfig:
    """
    Declarative configuration holder for SMS-EMOA settings.

    Examples:
        cfg = SMSEMOAConfig.default()
        cfg = SMSEMOAConfig().pop_size(100).crossover("sbx", prob=0.9).fixed()
    """

    def __init__(self) -> None:
        self._cfg: Dict[str, Any] = {}

    @classmethod
    def default(
        cls,
        pop_size: int = 100,
        n_var: int | None = None,
        engine: str = "numpy",
    ) -> "SMSEMOAConfigData":
        """Create a default SMS-EMOA configuration."""
        mut_prob = 1.0 / n_var if n_var else 0.1
        return (
            cls()
            .pop_size(pop_size)
            .crossover("sbx", prob=0.9, eta=20.0)
            .mutation("pm", prob=mut_prob, eta=20.0)
            .selection("tournament")
            .reference_point(adaptive=True)
            .engine(engine)
            .fixed()
        )

    def pop_size(self, value: int) -> "SMSEMOAConfig":
        self._cfg["pop_size"] = value
        return self

    def crossover(self, method: str, **kwargs) -> "SMSEMOAConfig":
        self._cfg["crossover"] = (method, kwargs)
        return self

    def mutation(self, method: str, **kwargs) -> "SMSEMOAConfig":
        self._cfg["mutation"] = (method, kwargs)
        return self

    def selection(self, method: str, **kwargs) -> "SMSEMOAConfig":
        self._cfg["selection"] = (method, kwargs)
        return self

    def reference_point(
        self,
        *,
        vector=None,
        offset: float = 0.1,
        adaptive: bool = True,
    ) -> "SMSEMOAConfig":
        self._cfg["reference_point"] = {
            "vector": vector,
            "offset": offset,
            "adaptive": adaptive,
        }
        return self

    def engine(self, value: str) -> "SMSEMOAConfig":
        self._cfg["engine"] = value
        return self

    def constraint_mode(self, value: str) -> "SMSEMOAConfig":
        self._cfg["constraint_mode"] = value
        return self

    def repair(self, method: str, **kwargs) -> "SMSEMOAConfig":
        self._cfg["repair"] = (method, kwargs)
        return self

    def initializer(self, method: str, **kwargs) -> "SMSEMOAConfig":
        self._cfg["initializer"] = {"type": method, **kwargs}
        return self

    def mutation_prob_factor(self, value: float) -> "SMSEMOAConfig":
        self._cfg["mutation_prob_factor"] = float(value)
        return self

    def track_genealogy(self, enabled: bool = True) -> "SMSEMOAConfig":
        self._cfg["track_genealogy"] = bool(enabled)
        return self

    def result_mode(self, value: str) -> "SMSEMOAConfig":
        self._cfg["result_mode"] = str(value)
        return self

    def external_archive(self, *, size: int, archive_type: str = "hypervolume") -> "SMSEMOAConfig":
        if size <= 0:
            raise ValueError("external archive size must be positive.")
        self._cfg["external_archive"] = {"size": int(size)}
        self._cfg["result_mode"] = "external_archive"
        self._cfg["archive_type"] = archive_type
        return self

    def archive_type(self, value: str) -> "SMSEMOAConfig":
        """Set archive pruning strategy: 'hypervolume' or 'crowding'."""
        self._cfg["archive_type"] = str(value)
        return self

    def archive(self, size: int) -> "SMSEMOAConfig":
        """
        Convenience alias to configure an external archive by size.
        A size <= 0 disables the archive.
        """
        if size <= 0:
            self._cfg["archive"] = {"size": 0}
            return self
        self._cfg["archive"] = {"size": int(size)}
        return self

    def fixed(self) -> SMSEMOAConfigData:
        _require_fields(
            self._cfg,
            ("pop_size", "crossover", "mutation", "selection", "engine"),
            "SMS-EMOA",
        )
        reference_point = self._cfg.get("reference_point", {"offset": 0.1, "adaptive": True})
        return SMSEMOAConfigData(
            pop_size=self._cfg["pop_size"],
            crossover=self._cfg["crossover"],
            mutation=self._cfg["mutation"],
            selection=self._cfg["selection"],
            reference_point=reference_point,
            engine=self._cfg["engine"],
            constraint_mode=self._cfg.get("constraint_mode", "feasibility"),
            repair=self._cfg.get("repair"),
            initializer=self._cfg.get("initializer"),
            mutation_prob_factor=self._cfg.get("mutation_prob_factor"),
            track_genealogy=bool(self._cfg.get("track_genealogy", False)),
            result_mode=self._cfg.get("result_mode"),
            archive_type=self._cfg.get("archive_type"),
            archive=self._cfg.get("archive", self._cfg.get("external_archive")),
        )
