"""SMS-EMOA configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vamos.archive import ExternalArchiveConfig
from .base import ConstraintModeStr, ResultMode, _SerializableConfig, _require_fields


@dataclass(frozen=True)
class SMSEMOAConfig(_SerializableConfig):
    pop_size: int
    crossover: tuple[str, dict[str, Any]]
    mutation: tuple[str, dict[str, Any]]
    selection: tuple[str, dict[str, Any]]
    reference_point: dict[str, Any]
    eliminate_duplicates: bool = False
    constraint_mode: ConstraintModeStr = "feasibility"
    repair: tuple[str, dict[str, Any]] | None = None
    initializer: dict[str, Any] | None = None
    mutation_prob_factor: float | None = None
    track_genealogy: bool = False
    result_mode: ResultMode | None = None
    external_archive: ExternalArchiveConfig | None = None

    @classmethod
    def default(
        cls,
        pop_size: int = 100,
        n_var: int | None = None,
    ) -> SMSEMOAConfig:
        """Create a default SMS-EMOA configuration."""
        mut_prob = 1.0 / n_var if n_var else 0.1
        return (
            cls.builder()
            .pop_size(pop_size)
            .crossover("sbx", prob=1.0, eta=20.0)
            .mutation("pm", prob=mut_prob, eta=20.0)
            .selection("random")
            .reference_point(adaptive=True)
            .build()
        )

    @classmethod
    def builder(cls) -> _SMSEMOAConfigBuilder:
        return _SMSEMOAConfigBuilder()


class _SMSEMOAConfigBuilder:
    """
    Declarative configuration holder for SMS-EMOA settings.

    Examples:
        cfg = SMSEMOAConfig.default()
        cfg = SMSEMOAConfig.builder().pop_size(100).crossover("sbx", prob=1.0).build()
    """

    def __init__(self) -> None:
        self._cfg: dict[str, Any] = {}

    def pop_size(self, value: int) -> _SMSEMOAConfigBuilder:
        self._cfg["pop_size"] = value
        return self

    def crossover(self, method: str, **kwargs: Any) -> _SMSEMOAConfigBuilder:
        self._cfg["crossover"] = (method, kwargs)
        return self

    def mutation(self, method: str, **kwargs: Any) -> _SMSEMOAConfigBuilder:
        self._cfg["mutation"] = (method, kwargs)
        return self

    def selection(self, method: str, **kwargs: Any) -> _SMSEMOAConfigBuilder:
        self._cfg["selection"] = (method, kwargs)
        return self

    def eliminate_duplicates(self, enabled: bool = True) -> _SMSEMOAConfigBuilder:
        self._cfg["eliminate_duplicates"] = bool(enabled)
        return self

    def reference_point(
        self,
        *,
        vector: Any = None,
        offset: float = 1.0,
        adaptive: bool = True,
    ) -> _SMSEMOAConfigBuilder:
        self._cfg["reference_point"] = {
            "vector": vector,
            "offset": offset,
            "adaptive": adaptive,
        }
        return self

    def constraint_mode(self, value: str) -> _SMSEMOAConfigBuilder:
        self._cfg["constraint_mode"] = value
        return self

    def repair(self, method: str, **kwargs: Any) -> _SMSEMOAConfigBuilder:
        self._cfg["repair"] = (method, kwargs)
        return self

    def initializer(self, method: str, **kwargs: Any) -> _SMSEMOAConfigBuilder:
        self._cfg["initializer"] = {"type": method, **kwargs}
        return self

    def mutation_prob_factor(self, value: float) -> _SMSEMOAConfigBuilder:
        self._cfg["mutation_prob_factor"] = float(value)
        return self

    def track_genealogy(self, enabled: bool = True) -> _SMSEMOAConfigBuilder:
        self._cfg["track_genealogy"] = bool(enabled)
        return self

    def result_mode(self, value: str) -> _SMSEMOAConfigBuilder:
        mode = str(value).strip().lower()
        if mode not in {"non_dominated", "population"}:
            raise ValueError("result_mode must be 'non_dominated' or 'population'.")
        self._cfg["result_mode"] = mode
        return self

    def external_archive(self, capacity: int | None = None, **kwargs: Any) -> _SMSEMOAConfigBuilder:
        """Configure an external archive.

        Args:
            capacity: Maximum number of solutions. ``None`` means unbounded.
            **kwargs: Forwarded to :class:`ExternalArchiveConfig`.
        """
        self._cfg["external_archive"] = ExternalArchiveConfig(capacity=capacity, **kwargs)
        return self

    def build(self) -> SMSEMOAConfig:
        _require_fields(
            self._cfg,
            ("pop_size", "crossover", "mutation", "selection"),
            "SMS-EMOA",
        )
        reference_point = self._cfg.get("reference_point", {"offset": 1.0, "adaptive": True})
        return SMSEMOAConfig(
            pop_size=self._cfg["pop_size"],
            crossover=self._cfg["crossover"],
            mutation=self._cfg["mutation"],
            selection=self._cfg["selection"],
            reference_point=reference_point,
            eliminate_duplicates=bool(self._cfg.get("eliminate_duplicates", False)),
            constraint_mode=self._cfg.get("constraint_mode", "feasibility"),
            repair=self._cfg.get("repair"),
            initializer=self._cfg.get("initializer"),
            mutation_prob_factor=self._cfg.get("mutation_prob_factor"),
            track_genealogy=bool(self._cfg.get("track_genealogy", False)),
            result_mode=self._cfg.get("result_mode", "non_dominated"),
            external_archive=self._cfg.get("external_archive"),
        )
