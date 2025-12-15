from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple


class _SerializableConfig:
    """Mixin to serialize dataclass configs."""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


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


@dataclass(frozen=True)
class SMSEMOAConfigData(_SerializableConfig):
    pop_size: int
    crossover: Tuple[str, Dict[str, Any]]
    mutation: Tuple[str, Dict[str, Any]]
    selection: Tuple[str, Dict[str, Any]]
    reference_point: Dict[str, Any]
    engine: str


@dataclass(frozen=True)
class NSGAIIIConfigData(_SerializableConfig):
    pop_size: int
    crossover: Tuple[str, Dict[str, Any]]
    mutation: Tuple[str, Dict[str, Any]]
    selection: Tuple[str, Dict[str, Any]]
    reference_directions: Dict[str, Optional[int | str]]
    engine: str


@dataclass(frozen=True)
class SPEA2ConfigData(_SerializableConfig):
    pop_size: int
    archive_size: int
    crossover: Tuple[str, Dict[str, Any]]
    mutation: Tuple[str, Dict[str, Any]]
    selection: Tuple[str, Dict[str, Any]]
    engine: str
    k_neighbors: Optional[int] = None
    repair: Optional[Tuple[str, Dict[str, Any]]] = None
    initializer: Optional[Dict[str, Any]] = None
    mutation_prob_factor: Optional[float] = None
    constraint_mode: str = "feasibility"


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


@dataclass(frozen=True)
class SMPSOConfigData(_SerializableConfig):
    pop_size: int
    archive_size: int
    mutation: Tuple[str, Dict[str, Any]]
    engine: str
    inertia: float = 0.5
    c1: float = 1.5
    c2: float = 1.5
    vmax_fraction: float = 0.5
    repair: Optional[Tuple[str, Dict[str, Any]]] = None
    initializer: Optional[Dict[str, Any]] = None
    constraint_mode: str = "feasibility"


def _require_fields(cfg: Dict[str, Any], fields: Tuple[str, ...], name: str) -> None:
    missing = [field for field in fields if field not in cfg]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{name} configuration missing required fields: {joined}")


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

        Examples:
            cfg = NSGAIIConfig.from_dict({
                "pop_size": 100,
                "crossover": ("sbx", {"prob": 0.9, "eta": 20}),
                "mutation": ("pm", {"prob": 0.1, "eta": 20}),
                "selection": "tournament",
                "survival": "rank_crowding",
                "engine": "numpy"
            })
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
        """
        Set constraint handling mode: 'feasibility' or 'none'/'penalty'.
        """
        self._cfg["constraint_mode"] = value
        return self

    def track_genealogy(self, enabled: bool = True) -> "NSGAIIConfig":
        self._cfg["track_genealogy"] = bool(enabled)
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
            result_mode=self._cfg.get("result_mode"),
            archive_type=self._cfg.get("archive_type"),
            constraint_mode=self._cfg.get("constraint_mode", "feasibility"),
            track_genealogy=bool(self._cfg.get("track_genealogy", False)),
        )


class MOEADConfig:
    """
    Declarative configuration holder for MOEA/D settings.
    Mirrors NSGA-II builder style so both algorithms can share patterns.

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
        """
        Create a default MOEA/D configuration with sensible defaults.

        Args:
            pop_size: Population size (default: 100)
            n_var: Number of variables (used for mutation prob = 1/n_var)
            engine: Backend engine (default: "numpy")

        Returns:
            Frozen MOEADConfigData ready to use
        """
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
        """
        Create configuration from a dictionary.

        Args:
            config: Dictionary with configuration keys

        Returns:
            Frozen MOEADConfigData
        """
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
        """
        Set constraint handling mode: 'feasibility' or 'penalty'.
        """
        self._cfg["constraint_mode"] = value
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
        )


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
            .selection("tournament")  # SMS-EMOA requires tournament selection
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
        )


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
        # divisions for Das-Dennis: choose based on n_obj
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
        )


class SPEA2Config:
    """
    Declarative configuration holder for SPEA2 settings.
    Mirrors NSGA-II builder style to enable reuse of variation configs.

    Examples:
        cfg = SPEA2Config.default()
        cfg = SPEA2Config().pop_size(100).archive_size(100).fixed()
    """

    def __init__(self) -> None:
        self._cfg: Dict[str, Any] = {}

    @classmethod
    def default(
        cls,
        pop_size: int = 100,
        n_var: int | None = None,
        engine: str = "numpy",
    ) -> "SPEA2ConfigData":
        """Create a default SPEA2 configuration."""
        mut_prob = 1.0 / n_var if n_var else 0.1
        return (
            cls()
            .pop_size(pop_size)
            .archive_size(pop_size)
            .crossover("sbx", prob=0.9, eta=20.0)
            .mutation("pm", prob=mut_prob, eta=20.0)
            .selection("tournament")
            .engine(engine)
            .fixed()
        )

    def pop_size(self, value: int) -> "SPEA2Config":
        self._cfg["pop_size"] = value
        return self

    def archive_size(self, value: int) -> "SPEA2Config":
        self._cfg["archive_size"] = value
        return self

    def crossover(self, method: str, **kwargs) -> "SPEA2Config":
        self._cfg["crossover"] = (method, kwargs)
        return self

    def mutation(self, method: str, **kwargs) -> "SPEA2Config":
        self._cfg["mutation"] = (method, kwargs)
        return self

    def selection(self, method: str, **kwargs) -> "SPEA2Config":
        self._cfg["selection"] = (method, kwargs)
        return self

    def engine(self, value: str) -> "SPEA2Config":
        self._cfg["engine"] = value
        return self

    def k_neighbors(self, value: int) -> "SPEA2Config":
        self._cfg["k_neighbors"] = value
        return self

    def repair(self, method: str, **kwargs) -> "SPEA2Config":
        self._cfg["repair"] = (method, kwargs)
        return self

    def initializer(self, method: str, **kwargs) -> "SPEA2Config":
        self._cfg["initializer"] = {"type": method, **kwargs}
        return self

    def mutation_prob_factor(self, value: float) -> "SPEA2Config":
        self._cfg["mutation_prob_factor"] = float(value)
        return self

    def constraint_mode(self, value: str) -> "SPEA2Config":
        self._cfg["constraint_mode"] = value
        return self

    def fixed(self) -> SPEA2ConfigData:
        _require_fields(
            self._cfg,
            ("pop_size", "archive_size", "crossover", "mutation", "selection", "engine"),
            "SPEA2",
        )
        return SPEA2ConfigData(
            pop_size=self._cfg["pop_size"],
            archive_size=self._cfg["archive_size"],
            crossover=self._cfg["crossover"],
            mutation=self._cfg["mutation"],
            selection=self._cfg["selection"],
            engine=self._cfg["engine"],
            k_neighbors=self._cfg.get("k_neighbors"),
            repair=self._cfg.get("repair"),
            initializer=self._cfg.get("initializer"),
            mutation_prob_factor=self._cfg.get("mutation_prob_factor"),
            constraint_mode=self._cfg.get("constraint_mode", "feasibility"),
        )


class IBEAConfig:
    """
    Declarative configuration holder for IBEA settings.
    """

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
        )


class SMPSOConfig:
    """
    Declarative configuration holder for SMPSO settings.
    """

    def __init__(self) -> None:
        self._cfg: Dict[str, Any] = {}

    def pop_size(self, value: int) -> "SMPSOConfig":
        self._cfg["pop_size"] = value
        return self

    def archive_size(self, value: int) -> "SMPSOConfig":
        self._cfg["archive_size"] = value
        return self

    def mutation(self, method: str, **kwargs) -> "SMPSOConfig":
        self._cfg["mutation"] = (method, kwargs)
        return self

    def engine(self, value: str) -> "SMPSOConfig":
        self._cfg["engine"] = value
        return self

    def inertia(self, value: float) -> "SMPSOConfig":
        self._cfg["inertia"] = value
        return self

    def c1(self, value: float) -> "SMPSOConfig":
        self._cfg["c1"] = value
        return self

    def c2(self, value: float) -> "SMPSOConfig":
        self._cfg["c2"] = value
        return self

    def vmax_fraction(self, value: float) -> "SMPSOConfig":
        self._cfg["vmax_fraction"] = value
        return self

    def repair(self, method: str, **kwargs) -> "SMPSOConfig":
        self._cfg["repair"] = (method, kwargs)
        return self

    def initializer(self, method: str, **kwargs) -> "SMPSOConfig":
        self._cfg["initializer"] = {"type": method, **kwargs}
        return self

    def constraint_mode(self, value: str) -> "SMPSOConfig":
        self._cfg["constraint_mode"] = value
        return self

    def fixed(self) -> SMPSOConfigData:
        _require_fields(
            self._cfg,
            ("pop_size", "archive_size", "mutation", "engine"),
            "SMPSO",
        )
        return SMPSOConfigData(
            pop_size=self._cfg["pop_size"],
            archive_size=self._cfg["archive_size"],
            mutation=self._cfg["mutation"],
            engine=self._cfg["engine"],
            inertia=float(self._cfg.get("inertia", 0.5)),
            c1=float(self._cfg.get("c1", 1.5)),
            c2=float(self._cfg.get("c2", 1.5)),
            vmax_fraction=float(self._cfg.get("vmax_fraction", 0.5)),
            repair=self._cfg.get("repair"),
            initializer=self._cfg.get("initializer"),
            constraint_mode=self._cfg.get("constraint_mode", "feasibility"),
        )
