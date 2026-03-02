"""AGE-MOEA configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vamos.engine.archive import ExternalArchiveConfig
from vamos.engine.archive.bounded_archive import PrunePolicy
from vamos.foundation.encoding import normalize_encoding

from .base import ConstraintModeStr, ResultMode, _default_operators_for_encoding, _require_fields, _SerializableConfig, _validate_operators
from .types import CrossoverName, InitializerName, MutationName, RepairConfigValue, RepairName


class _AGEMOEAConfigBuilder:
    """
    Fluent builder for AGE-MOEA configs.
    """

    def __init__(self) -> None:
        self._cfg: dict[str, Any] = {}

    def pop_size(self, value: int) -> _AGEMOEAConfigBuilder:
        self._cfg["pop_size"] = value
        return self

    def crossover(self, method: CrossoverName | str, **kwargs: Any) -> _AGEMOEAConfigBuilder:
        self._cfg["crossover"] = (method, kwargs)
        return self

    def mutation(self, method: MutationName | str, **kwargs: Any) -> _AGEMOEAConfigBuilder:
        self._cfg["mutation"] = (method, kwargs)
        return self

    def repair(self, method: RepairName | str, **kwargs: Any) -> _AGEMOEAConfigBuilder:
        self._cfg["repair"] = (method, kwargs)
        return self

    def initializer(self, method: InitializerName | str, **kwargs: Any) -> _AGEMOEAConfigBuilder:
        self._cfg["initializer"] = {"type": method, **kwargs}
        return self

    def mutation_prob_factor(self, value: float) -> _AGEMOEAConfigBuilder:
        self._cfg["mutation_prob_factor"] = float(value)
        return self

    def constraint_mode(self, value: ConstraintModeStr | str) -> _AGEMOEAConfigBuilder:
        self._cfg["constraint_mode"] = value
        return self

    def track_genealogy(self, enabled: bool = True) -> _AGEMOEAConfigBuilder:
        self._cfg["track_genealogy"] = bool(enabled)
        return self

    def result_mode(self, value: ResultMode | str) -> _AGEMOEAConfigBuilder:
        mode = str(value).strip().lower()
        if mode not in {"non_dominated", "population"}:
            raise ValueError("result_mode must be 'non_dominated' or 'population'.")
        self._cfg["result_mode"] = mode
        return self

    def external_archive(
        self,
        capacity: int | None = None,
        pruning: PrunePolicy = "crowding",
    ) -> _AGEMOEAConfigBuilder:
        """Configure an external archive.

        Args:
            capacity: Maximum number of solutions. ``None`` means unbounded.
            pruning: Strategy used when bounded archive exceeds capacity.
        """
        self._cfg["external_archive"] = ExternalArchiveConfig(capacity=capacity, pruning=pruning)
        return self

    def build(self) -> AGEMOEAConfig:
        _require_fields(
            self._cfg,
            ("pop_size", "crossover", "mutation"),
            "AGE-MOEA",
        )
        _validate_operators(self._cfg)
        return AGEMOEAConfig(
            pop_size=self._cfg["pop_size"],
            crossover=self._cfg["crossover"],
            mutation=self._cfg["mutation"],
            repair=self._cfg.get("repair", "auto"),
            initializer=self._cfg.get("initializer"),
            mutation_prob_factor=self._cfg.get("mutation_prob_factor"),
            constraint_mode=self._cfg.get("constraint_mode", "feasibility"),
            track_genealogy=bool(self._cfg.get("track_genealogy", False)),
            result_mode=self._cfg.get("result_mode", "non_dominated"),
            external_archive=self._cfg.get("external_archive"),
        )


@dataclass(frozen=True, repr=False)
class AGEMOEAConfig(_SerializableConfig):
    pop_size: int
    crossover: tuple[str, dict[str, Any]]
    mutation: tuple[str, dict[str, Any]]
    repair: RepairConfigValue = "auto"
    initializer: dict[str, Any] | None = None
    mutation_prob_factor: float | None = None
    constraint_mode: ConstraintModeStr = "feasibility"
    track_genealogy: bool = False
    result_mode: ResultMode | None = None
    external_archive: ExternalArchiveConfig | None = None

    @classmethod
    def default(
        cls,
        pop_size: int = 100,
        n_var: int | None = None,
        encoding: str | None = None,
    ) -> AGEMOEAConfig:
        """Create a default AGE-MOEA configuration.

        Args:
            pop_size: Population size (default: 100)
            n_var: Number of variables (for mutation prob)
            encoding: Problem encoding. If omitted, defaults to "real".
        """
        mut_prob = 1.0 / n_var if n_var else 0.1
        normalized = normalize_encoding(encoding or "real")
        cx, mt = _default_operators_for_encoding(normalized, mut_prob)
        builder = cls.builder().pop_size(pop_size).crossover(cx[0], **cx[1]).mutation(mt[0], **mt[1])
        if normalized == "real":
            builder = builder.repair("clip")
        return builder.build()

    @classmethod
    def builder(cls) -> _AGEMOEAConfigBuilder:
        return _AGEMOEAConfigBuilder()
