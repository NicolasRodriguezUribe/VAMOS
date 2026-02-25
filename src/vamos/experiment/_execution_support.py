from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass as _dataclass
from importlib.resources import as_file
from typing import Any

import numpy as np

from vamos.engine.config.variation import VariationConfig
from vamos.foundation.observer import Observer, RunContext
from vamos.resources import weight_path


@_dataclass
class VariationConfigs:
    """Bundles per-algorithm variation configurations into a single object."""

    nsgaii: VariationConfig | None = None
    moead: VariationConfig | None = None
    smsemoa: VariationConfig | None = None
    nsgaiii: VariationConfig | None = None
    spea2: VariationConfig | None = None
    ibea: VariationConfig | None = None
    smpso: VariationConfig | None = None
    agemoea: VariationConfig | None = None
    rvea: VariationConfig | None = None

    @classmethod
    def from_namespace(cls, args: Namespace) -> VariationConfigs:
        return cls(
            nsgaii=getattr(args, "nsgaii_variation", None),
            moead=getattr(args, "moead_variation", None),
            smsemoa=getattr(args, "smsemoa_variation", None),
            nsgaiii=getattr(args, "nsgaiii_variation", None),
            spea2=getattr(args, "spea2_variation", None),
            ibea=getattr(args, "ibea_variation", None),
            smpso=getattr(args, "smpso_variation", None),
            agemoea=getattr(args, "agemoea_variation", None),
            rvea=getattr(args, "rvea_variation", None),
        )

    def as_storage_dict(self) -> dict[str, VariationConfig | None]:
        return {
            "nsgaii_variation": self.nsgaii,
            "moead_variation": self.moead,
            "smsemoa_variation": self.smsemoa,
            "nsgaiii_variation": self.nsgaiii,
            "spea2_variation": self.spea2,
            "ibea_variation": self.ibea,
            "smpso_variation": self.smpso,
            "agemoea_variation": self.agemoea,
            "rvea_variation": self.rvea,
        }


class CompositeObserver(Observer):
    """Fans out observer events to all registered observers."""

    def __init__(self, observers: list[Observer]):
        self.observers = [o for o in observers if o is not None]

    def on_start(self, ctx: RunContext) -> None:
        for obs in self.observers:
            obs.on_start(ctx)

    def on_generation(
        self,
        generation: int,
        F: np.ndarray | None = None,
        X: np.ndarray | None = None,
        stats: dict[str, Any] | None = None,
    ) -> None:
        for obs in self.observers:
            obs.on_generation(generation, F, X, stats)

    def on_end(
        self,
        final_F: np.ndarray | None = None,
        final_stats: dict[str, Any] | None = None,
    ) -> None:
        for obs in self.observers:
            obs.on_end(final_F, final_stats)

    def should_stop(self) -> bool:
        for obs in self.observers:
            if hasattr(obs, "should_stop") and obs.should_stop():
                return True
        return False


class LiveVizAdapter:
    """Bridge algorithm live-viz callbacks to observer on_generation events."""

    def __init__(self, observer: CompositeObserver):
        self._observer = observer

    def on_start(self, ctx: RunContext) -> None:
        return None

    def on_generation(
        self,
        generation: int,
        F: np.ndarray | None = None,
        X: np.ndarray | None = None,
        stats: dict[str, Any] | None = None,
    ) -> None:
        self._observer.on_generation(generation, F, X, stats)

    def on_end(
        self,
        final_F: np.ndarray | None = None,
        final_stats: dict[str, Any] | None = None,
    ) -> None:
        return None

    def should_stop(self) -> bool:
        return self._observer.should_stop()


def default_weight_path(problem_name: str, n_obj: int, pop_size: int) -> str:
    filename = f"{problem_name}_nobj{n_obj}_pop{pop_size}.csv"
    try:
        with as_file(weight_path(filename)) as p:
            return str(p)
    except Exception:
        return str(weight_path("zdt1problem_2obj_pop100.csv")) if "zdt1" in filename else str(weight_path(filename))
