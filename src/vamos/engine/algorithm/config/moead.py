"""MOEA/D configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vamos.foundation.data import weight_path

from .base import _SerializableConfig, _require_fields


@dataclass(frozen=True)
class MOEADConfig(_SerializableConfig):
    pop_size: int
    batch_size: int
    neighbor_size: int
    delta: float
    replace_limit: int
    crossover: tuple[str, dict[str, Any]]
    mutation: tuple[str, dict[str, Any]]
    aggregation: tuple[str, dict[str, Any]]
    weight_vectors: dict[str, int | str | None] | None
    constraint_mode: str = "feasibility"
    repair: tuple[str, dict[str, Any]] | None = None
    initializer: dict[str, Any] | None = None
    mutation_prob_factor: float | None = None
    use_numba_variation: bool | None = None
    track_genealogy: bool = False
    result_mode: str | None = None
    archive_type: str | None = None
    archive: dict[str, Any] | None = None

    @classmethod
    def default(
        cls,
        pop_size: int | None = None,
        n_var: int | None = None,
        n_obj: int = 3,
    ) -> MOEADConfig:
        """Create a default MOEA/D configuration with sensible defaults."""
        if pop_size is None:
            pop_size = 91 if n_obj == 3 else 100
        mut_prob = 1.0 / n_var if n_var else 0.1
        weights_dir = weight_path("W3D_91.dat").parent
        return (
            cls.builder()
            .pop_size(pop_size)
            .batch_size(1)
            .neighbor_size(20)
            .delta(0.9)
            .replace_limit(2)
            .crossover("de", cr=1.0, f=0.5)
            .mutation("pm", prob=mut_prob, eta=20.0)
            .aggregation("pbi", theta=5.0)
            .weight_vectors(path=str(weights_dir))
            .build()
        )

    @classmethod
    def builder(cls) -> _MOEADConfigBuilder:
        return _MOEADConfigBuilder()


class _MOEADConfigBuilder:
    """
    Declarative configuration holder for MOEA/D settings.

    Examples:
        # Fluent builder
        cfg = MOEADConfig.builder().pop_size(100).neighbor_size(20).build()

        # Quick default configuration
        cfg = MOEADConfig.default()
    """

    def __init__(self) -> None:
        self._cfg: dict[str, Any] = {}

    def pop_size(self, value: int) -> _MOEADConfigBuilder:
        self._cfg["pop_size"] = value
        return self

    def batch_size(self, value: int) -> _MOEADConfigBuilder:
        self._cfg["batch_size"] = value
        return self

    def neighbor_size(self, value: int) -> _MOEADConfigBuilder:
        self._cfg["neighbor_size"] = value
        return self

    def delta(self, value: float) -> _MOEADConfigBuilder:
        self._cfg["delta"] = value
        return self

    def replace_limit(self, value: int) -> _MOEADConfigBuilder:
        self._cfg["replace_limit"] = value
        return self

    def crossover(self, method: str, **kwargs: Any) -> _MOEADConfigBuilder:
        self._cfg["crossover"] = (method, kwargs)
        return self

    def mutation(self, method: str, **kwargs: Any) -> _MOEADConfigBuilder:
        self._cfg["mutation"] = (method, kwargs)
        return self

    def aggregation(self, method: str, **kwargs: Any) -> _MOEADConfigBuilder:
        self._cfg["aggregation"] = (method, kwargs)
        return self

    def weight_vectors(self, *, path: str | None = None, divisions: int | None = None) -> _MOEADConfigBuilder:
        self._cfg["weight_vectors"] = {"path": path, "divisions": divisions}
        return self

    def constraint_mode(self, value: str) -> _MOEADConfigBuilder:
        self._cfg["constraint_mode"] = value
        return self

    def repair(self, method: str, **kwargs: Any) -> _MOEADConfigBuilder:
        self._cfg["repair"] = (method, kwargs)
        return self

    def initializer(self, method: str, **kwargs: Any) -> _MOEADConfigBuilder:
        self._cfg["initializer"] = {"type": method, **kwargs}
        return self

    def mutation_prob_factor(self, value: float) -> _MOEADConfigBuilder:
        self._cfg["mutation_prob_factor"] = float(value)
        return self

    def use_numba_variation(self, enabled: bool = True) -> _MOEADConfigBuilder:
        self._cfg["use_numba_variation"] = bool(enabled)
        return self

    def track_genealogy(self, enabled: bool = True) -> _MOEADConfigBuilder:
        self._cfg["track_genealogy"] = bool(enabled)
        return self

    def result_mode(self, value: str) -> _MOEADConfigBuilder:
        mode = str(value).strip().lower()
        if mode not in {"non_dominated", "population"}:
            raise ValueError("result_mode must be 'non_dominated' or 'population'.")
        self._cfg["result_mode"] = mode
        return self

    def archive_type(self, value: str) -> _MOEADConfigBuilder:
        """Set archive pruning strategy: 'hypervolume' or 'crowding'."""
        self._cfg["archive_type"] = str(value)
        return self

    def archive(self, size: int, **kwargs: Any) -> _MOEADConfigBuilder:
        """
        Configure an external archive.

        Args:
            size: Archive size (required). <= 0 disables the archive.
            **kwargs: Optional configuration:
                - archive_type: "size_cap", "epsilon_grid", "hvc_prune", "hybrid"
                - prune_policy: "crowding", "hv_contrib", "random"
                - epsilon: Grid epsilon for epsilon_grid/hybrid types

        Notes:
            This method configures archive storage only. It does not change
            ``result_mode``.
        """
        if size <= 0:
            self._cfg["archive"] = {"size": 0}
            return self
        archive_cfg = {"size": int(size), **kwargs}
        self._cfg["archive"] = archive_cfg
        return self

    def build(self) -> MOEADConfig:
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
            ),
            "MOEA/D",
        )
        return MOEADConfig(
            pop_size=self._cfg["pop_size"],
            batch_size=int(self._cfg.get("batch_size", 1)),
            neighbor_size=self._cfg["neighbor_size"],
            delta=self._cfg["delta"],
            replace_limit=self._cfg["replace_limit"],
            crossover=self._cfg["crossover"],
            mutation=self._cfg["mutation"],
            aggregation=self._cfg["aggregation"],
            weight_vectors=self._cfg.get("weight_vectors"),
            constraint_mode=self._cfg.get("constraint_mode", "feasibility"),
            repair=self._cfg.get("repair"),
            initializer=self._cfg.get("initializer"),
            mutation_prob_factor=self._cfg.get("mutation_prob_factor"),
            use_numba_variation=self._cfg.get("use_numba_variation"),
            track_genealogy=bool(self._cfg.get("track_genealogy", False)),
            result_mode=self._cfg.get("result_mode", "non_dominated"),
            archive_type=self._cfg.get("archive_type"),
            archive=self._cfg.get("archive"),
        )
