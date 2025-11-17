from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple


class _SerializableConfig:
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


def _require_fields(cfg: Dict[str, Any], fields: Tuple[str, ...], name: str) -> None:
    missing = [field for field in fields if field not in cfg]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{name} configuration missing required fields: {joined}")


class NSGAIIConfig:
    """
    Declarative configuration holder for NSGA-II.
    """

    def __init__(self):
        self._cfg: Dict[str, Any] = {}

    def pop_size(self, value: int):
        self._cfg["pop_size"] = value
        return self

    def crossover(self, method: str, **kwargs):
        self._cfg["crossover"] = (method, kwargs)
        return self

    def mutation(self, method: str, **kwargs):
        self._cfg["mutation"] = (method, kwargs)
        return self

    def selection(self, method: str, **kwargs):
        self._cfg["selection"] = (method, kwargs)
        return self

    def survival(self, method: str):
        self._cfg["survival"] = method
        return self

    def engine(self, value: str):
        self._cfg["engine"] = value
        return self

    def fixed(self) -> NSGAIIConfigData:
        _require_fields(
            self._cfg,
            ("pop_size", "crossover", "mutation", "selection", "survival", "engine"),
            "NSGA-II",
        )
        return NSGAIIConfigData(
            pop_size=self._cfg["pop_size"],
            crossover=self._cfg["crossover"],
            mutation=self._cfg["mutation"],
            selection=self._cfg["selection"],
            survival=self._cfg["survival"],
            engine=self._cfg["engine"],
        )


class MOEADConfig:
    """
    Declarative configuration holder for MOEA/D settings.
    Mirrors NSGA-II builder style so both algorithms can share patterns.
    """

    def __init__(self):
        self._cfg: Dict[str, Any] = {}

    def pop_size(self, value: int):
        self._cfg["pop_size"] = value
        return self

    def neighbor_size(self, value: int):
        self._cfg["neighbor_size"] = value
        return self

    def delta(self, value: float):
        self._cfg["delta"] = value
        return self

    def replace_limit(self, value: int):
        self._cfg["replace_limit"] = value
        return self

    def crossover(self, method: str, **kwargs):
        self._cfg["crossover"] = (method, kwargs)
        return self

    def mutation(self, method: str, **kwargs):
        self._cfg["mutation"] = (method, kwargs)
        return self

    def aggregation(self, method: str, **kwargs):
        self._cfg["aggregation"] = (method, kwargs)
        return self

    def weight_vectors(
        self, *, path: Optional[str] = None, divisions: Optional[int] = None
    ):
        self._cfg["weight_vectors"] = {"path": path, "divisions": divisions}
        return self

    def engine(self, value: str):
        self._cfg["engine"] = value
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
        )


class SMSEMOAConfig:
    """
    Declarative configuration holder for SMS-EMOA settings.
    """

    def __init__(self):
        self._cfg: Dict[str, Any] = {}

    def pop_size(self, value: int):
        self._cfg["pop_size"] = value
        return self

    def crossover(self, method: str, **kwargs):
        self._cfg["crossover"] = (method, kwargs)
        return self

    def mutation(self, method: str, **kwargs):
        self._cfg["mutation"] = (method, kwargs)
        return self

    def selection(self, method: str, **kwargs):
        self._cfg["selection"] = (method, kwargs)
        return self

    def reference_point(
        self,
        *,
        vector=None,
        offset: float = 0.1,
        adaptive: bool = True,
    ):
        self._cfg["reference_point"] = {
            "vector": vector,
            "offset": offset,
            "adaptive": adaptive,
        }
        return self

    def engine(self, value: str):
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
    """

    def __init__(self):
        self._cfg: Dict[str, Any] = {}

    def pop_size(self, value: int):
        self._cfg["pop_size"] = value
        return self

    def crossover(self, method: str, **kwargs):
        self._cfg["crossover"] = (method, kwargs)
        return self

    def mutation(self, method: str, **kwargs):
        self._cfg["mutation"] = (method, kwargs)
        return self

    def selection(self, method: str, **kwargs):
        self._cfg["selection"] = (method, kwargs)
        return self

    def reference_directions(
        self,
        *,
        path: Optional[str] = None,
        divisions: Optional[int] = None,
    ):
        self._cfg["reference_directions"] = {"path": path, "divisions": divisions}
        return self

    def engine(self, value: str):
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
