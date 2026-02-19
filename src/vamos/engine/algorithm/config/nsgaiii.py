"""NSGA-III configuration."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any

from vamos.archive import ExternalArchiveConfig

from .base import ConstraintModeStr, ResultMode, _require_fields, _SerializableConfig


@dataclass(frozen=True)
class NSGAIIIConfig(_SerializableConfig):
    pop_size: int
    crossover: tuple[str, dict[str, Any]]
    mutation: tuple[str, dict[str, Any]]
    selection: tuple[str, dict[str, Any]]
    reference_directions: dict[str, int | str | None]
    enforce_ref_dirs: bool = True
    pop_size_auto: bool = False
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
        pop_size: int | None = None,
        n_var: int | None = None,
        n_obj: int = 3,
    ) -> NSGAIIIConfig:
        """
        Create a default NSGA-III configuration.

        Args:
            pop_size: Population size (default: matches reference directions)
            n_var: Number of variables (for mutation prob)
            n_obj: Number of objectives (for reference directions)
        """
        mut_prob = 1.0 / n_var if n_var else 0.1
        divisions = 12 if n_obj == 3 else 6
        if pop_size is None:
            pop_size = comb(divisions + n_obj - 1, n_obj - 1)
        return (
            cls.builder()
            .pop_size(pop_size)
            .crossover("sbx", prob=1.0, eta=30.0)
            .mutation("pm", prob=mut_prob, eta=20.0)
            .selection("tournament")
            .reference_directions(divisions=divisions)
            .pop_size_auto(True)
            .build()
        )

    @classmethod
    def builder(cls) -> _NSGAIIIConfigBuilder:
        return _NSGAIIIConfigBuilder()


class _NSGAIIIConfigBuilder:
    """
    Declarative configuration holder for NSGA-III settings.

    Examples:
        cfg = NSGAIIIConfig.default(n_obj=3)
        cfg = NSGAIIIConfig.builder().pop_size(92).crossover("sbx", prob=1.0).build()
    """

    def __init__(self) -> None:
        self._cfg: dict[str, Any] = {}

    def pop_size(self, value: int) -> _NSGAIIIConfigBuilder:
        self._cfg["pop_size"] = value
        return self

    def crossover(self, method: str, **kwargs: Any) -> _NSGAIIIConfigBuilder:
        self._cfg["crossover"] = (method, kwargs)
        return self

    def mutation(self, method: str, **kwargs: Any) -> _NSGAIIIConfigBuilder:
        self._cfg["mutation"] = (method, kwargs)
        return self

    def selection(self, method: str, **kwargs: Any) -> _NSGAIIIConfigBuilder:
        self._cfg["selection"] = (method, kwargs)
        return self

    def reference_directions(
        self,
        *,
        path: str | None = None,
        divisions: int | None = None,
    ) -> _NSGAIIIConfigBuilder:
        self._cfg["reference_directions"] = {"path": path, "divisions": divisions}
        return self

    def enforce_ref_dirs(self, enabled: bool = True) -> _NSGAIIIConfigBuilder:
        self._cfg["enforce_ref_dirs"] = bool(enabled)
        return self

    def pop_size_auto(self, enabled: bool = True) -> _NSGAIIIConfigBuilder:
        self._cfg["pop_size_auto"] = bool(enabled)
        return self

    def constraint_mode(self, value: str) -> _NSGAIIIConfigBuilder:
        self._cfg["constraint_mode"] = value
        return self

    def repair(self, method: str, **kwargs: Any) -> _NSGAIIIConfigBuilder:
        self._cfg["repair"] = (method, kwargs)
        return self

    def initializer(self, method: str, **kwargs: Any) -> _NSGAIIIConfigBuilder:
        self._cfg["initializer"] = {"type": method, **kwargs}
        return self

    def mutation_prob_factor(self, value: float) -> _NSGAIIIConfigBuilder:
        self._cfg["mutation_prob_factor"] = float(value)
        return self

    def track_genealogy(self, enabled: bool = True) -> _NSGAIIIConfigBuilder:
        self._cfg["track_genealogy"] = bool(enabled)
        return self

    def result_mode(self, value: str) -> _NSGAIIIConfigBuilder:
        self._cfg["result_mode"] = str(value)
        return self

    def external_archive(self, capacity: int | None = None, **kwargs: Any) -> _NSGAIIIConfigBuilder:
        """Configure an external archive.

        Args:
            capacity: Maximum number of solutions. ``None`` means unbounded.
            **kwargs: Forwarded to :class:`ExternalArchiveConfig`.
        """
        self._cfg["external_archive"] = ExternalArchiveConfig(capacity=capacity, **kwargs)
        return self

    def build(self) -> NSGAIIIConfig:
        _require_fields(
            self._cfg,
            ("pop_size", "crossover", "mutation", "selection"),
            "NSGA-III",
        )
        ref_dirs = self._cfg.get("reference_directions", {})
        return NSGAIIIConfig(
            pop_size=self._cfg["pop_size"],
            crossover=self._cfg["crossover"],
            mutation=self._cfg["mutation"],
            selection=self._cfg["selection"],
            reference_directions=ref_dirs,
            enforce_ref_dirs=bool(self._cfg.get("enforce_ref_dirs", True)),
            pop_size_auto=bool(self._cfg.get("pop_size_auto", False)),
            constraint_mode=self._cfg.get("constraint_mode", "feasibility"),
            repair=self._cfg.get("repair"),
            initializer=self._cfg.get("initializer"),
            mutation_prob_factor=self._cfg.get("mutation_prob_factor"),
            track_genealogy=bool(self._cfg.get("track_genealogy", False)),
            result_mode=self._cfg.get("result_mode", "population"),
            external_archive=self._cfg.get("external_archive"),
        )
