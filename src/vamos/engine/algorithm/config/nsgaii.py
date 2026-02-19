"""NSGA-II configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vamos.archive import ExternalArchiveConfig
from vamos.foundation.encoding import normalize_encoding
from .base import ConstraintModeStr, LiveCallbackMode, ResultMode, _SerializableConfig, _require_fields


class _NSGAIIConfigBuilder:
    """
    Fluent builder for NSGA-II configs.
    """

    def __init__(self) -> None:
        self._cfg: dict[str, Any] = {}

    def pop_size(self, value: int) -> _NSGAIIConfigBuilder:
        self._cfg["pop_size"] = value
        return self

    def crossover(
        self,
        method: str | tuple[str, dict[str, Any]],
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> _NSGAIIConfigBuilder:
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
    ) -> _NSGAIIConfigBuilder:
        if isinstance(method, tuple) and params is None and not kwargs:
            method, params = method
        cfg_kwargs = params or kwargs
        self._cfg["mutation"] = (method, cfg_kwargs)
        return self

    def offspring_size(self, value: int) -> _NSGAIIConfigBuilder:
        if value <= 0:
            raise ValueError("offspring size must be positive.")
        self._cfg["offspring_size"] = value
        return self

    def steady_state(self, enabled: bool = True) -> _NSGAIIConfigBuilder:
        """Enable steady-state mode (incremental replacement)."""
        self._cfg["steady_state"] = bool(enabled)
        return self

    def replacement_size(self, value: int) -> _NSGAIIConfigBuilder:
        if value <= 0:
            raise ValueError("replacement size must be positive.")
        self._cfg["replacement_size"] = value
        return self

    def repair(self, method: str, **kwargs: Any) -> _NSGAIIConfigBuilder:
        self._cfg["repair"] = (method, kwargs)
        return self

    def selection(self, method: str, **kwargs: Any) -> _NSGAIIConfigBuilder:
        self._cfg["selection"] = (method, kwargs)
        return self

    def initializer(self, method: str, **kwargs: Any) -> _NSGAIIConfigBuilder:
        self._cfg["initializer"] = {"type": method, **kwargs}
        return self

    def mutation_prob_factor(self, value: float) -> _NSGAIIConfigBuilder:
        self._cfg["mutation_prob_factor"] = float(value)
        return self

    def result_mode(self, value: str) -> _NSGAIIConfigBuilder:
        """Set result payload mode: ``non_dominated`` or ``population``."""
        mode = str(value).strip().lower()
        if mode not in {"non_dominated", "population"}:
            raise ValueError("result_mode must be 'non_dominated' or 'population'.")
        self._cfg["result_mode"] = mode
        return self

    def external_archive(
        self,
        capacity: int | None = None,
        **kwargs: Any,
    ) -> _NSGAIIConfigBuilder:
        """Configure an external archive.

        Args:
            capacity: Maximum number of solutions. ``None`` means unbounded.
            **kwargs: Forwarded to :class:`ExternalArchiveConfig`
                (e.g. ``pruning``, ``archive_type``, ``epsilon``).
        """
        self._cfg["external_archive"] = ExternalArchiveConfig(capacity=capacity, **kwargs)
        return self

    def constraint_mode(self, value: str) -> _NSGAIIConfigBuilder:
        """Set constraint handling mode: 'feasibility' or 'none'/'penalty'."""
        self._cfg["constraint_mode"] = value
        return self

    def track_genealogy(self, enabled: bool = True) -> _NSGAIIConfigBuilder:
        self._cfg["track_genealogy"] = bool(enabled)
        return self

    def adaptive_operator_selection(self, config: dict[str, Any] | None) -> _NSGAIIConfigBuilder:
        if config is None:
            self._cfg["adaptive_operator_selection"] = None
        else:
            self._cfg["adaptive_operator_selection"] = dict(config)
        return self

    def immigration(self, config: dict[str, Any] | None) -> _NSGAIIConfigBuilder:
        if config is None:
            self._cfg["immigration"] = None
        else:
            self._cfg["immigration"] = dict(config)
        return self

    def parent_selection_filter(self, fn: Any | None) -> _NSGAIIConfigBuilder:
        self._cfg["parent_selection_filter"] = fn
        return self

    def live_callback_mode(self, mode: str) -> _NSGAIIConfigBuilder:
        self._cfg["live_callback_mode"] = str(mode)
        return self

    def generation_callback(
        self,
        fn: Any | None,
        *,
        copy_arrays: bool = True,
    ) -> _NSGAIIConfigBuilder:
        self._cfg["generation_callback"] = fn
        self._cfg["generation_callback_copy"] = bool(copy_arrays)
        return self

    def build(self) -> NSGAIIConfig:
        _require_fields(
            self._cfg,
            ("crossover", "mutation"),
            "NSGA-II",
        )
        pop_size = int(self._cfg.get("pop_size", 100))
        selection = self._cfg.get("selection", ("tournament", {}))
        return NSGAIIConfig(
            pop_size=pop_size,
            crossover=self._cfg["crossover"],
            mutation=self._cfg["mutation"],
            selection=selection,
            offspring_size=self._cfg.get("offspring_size"),
            steady_state=bool(self._cfg.get("steady_state", False)),
            replacement_size=self._cfg.get("replacement_size"),
            repair=self._cfg.get("repair"),
            external_archive=self._cfg.get("external_archive"),
            initializer=self._cfg.get("initializer"),
            mutation_prob_factor=self._cfg.get("mutation_prob_factor"),
            result_mode=self._cfg.get("result_mode", "non_dominated"),
            constraint_mode=self._cfg.get("constraint_mode", "feasibility"),
            track_genealogy=bool(self._cfg.get("track_genealogy", False)),
            adaptive_operator_selection=self._cfg.get("adaptive_operator_selection"),
            immigration=self._cfg.get("immigration"),
            parent_selection_filter=self._cfg.get("parent_selection_filter"),
            live_callback_mode=self._cfg.get("live_callback_mode", "nd_only"),
            generation_callback=self._cfg.get("generation_callback"),
            generation_callback_copy=bool(self._cfg.get("generation_callback_copy", True)),
        )


@dataclass(frozen=True)
class NSGAIIConfig(_SerializableConfig):
    pop_size: int
    crossover: tuple[str, dict[str, Any]]
    mutation: tuple[str, dict[str, Any]]
    selection: tuple[str, dict[str, Any]]
    offspring_size: int | None = None
    steady_state: bool = False
    replacement_size: int | None = None
    repair: tuple[str, dict[str, Any]] | None = None
    external_archive: ExternalArchiveConfig | None = None
    initializer: dict[str, Any] | None = None
    mutation_prob_factor: float | None = None
    result_mode: ResultMode | None = None
    constraint_mode: ConstraintModeStr = "feasibility"
    track_genealogy: bool = False
    adaptive_operator_selection: dict[str, Any] | None = None
    immigration: dict[str, Any] | None = None
    parent_selection_filter: Any | None = None
    live_callback_mode: LiveCallbackMode = "nd_only"
    generation_callback: Any | None = None
    generation_callback_copy: bool = True

    @classmethod
    def default(
        cls,
        pop_size: int = 100,
        n_var: int | None = None,
        encoding: str | None = None,
    ) -> NSGAIIConfig:
        """
        Create a default NSGA-II configuration with sensible defaults.

        Args:
            pop_size: Population size (default: 100)
            n_var: Number of variables (used for mutation prob = 1/n_var)
            encoding: Problem encoding. If omitted, defaults to "real".
        Returns:
            Frozen NSGAIIConfig ready to use
        """
        normalized = normalize_encoding(encoding, default="real")
        mut_prob = 1.0 / n_var if n_var else 0.1
        builder = cls.builder().pop_size(pop_size).selection("tournament")

        if normalized == "permutation":
            return builder.crossover("ox").mutation("swap").build()
        if normalized == "binary":
            return builder.crossover("uniform", prob=0.9).mutation("bitflip", prob=mut_prob).build()
        if normalized == "integer":
            return builder.crossover("sbx", prob=0.9, eta=20.0).mutation("pm", prob=mut_prob, eta=20.0).build()
        if normalized == "mixed":
            return builder.crossover("mixed", prob=0.9).mutation("mixed", prob=mut_prob).build()

        return builder.crossover("sbx", prob=1.0, eta=20.0).mutation("pm", prob=mut_prob, eta=20.0).build()

    @classmethod
    def builder(cls) -> _NSGAIIConfigBuilder:
        return _NSGAIIConfigBuilder()
