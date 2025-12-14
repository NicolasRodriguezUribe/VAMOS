from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class Real:
    """Real-valued hyperparameter in [low, high]."""

    low: float
    high: float
    log: bool = False

    def sample(self, rng: np.random.Generator) -> float:
        if self.log:
            low = math.log(self.low)
            high = math.log(self.high)
            v = rng.uniform(low, high)
            return float(math.exp(v))
        return float(rng.uniform(self.low, self.high))


@dataclass
class Int:
    """Integer hyperparameter in [low, high] (inclusive)."""

    low: int
    high: int
    log: bool = False

    def sample(self, rng: np.random.Generator) -> int:
        if self.log:
            low = math.log(self.low)
            high = math.log(self.high)
            v = rng.uniform(low, high)
            return int(round(math.exp(v)))
        return int(rng.integers(self.low, self.high + 1))


@dataclass
class Categorical:
    """Categorical hyperparameter."""

    choices: Sequence[Any]

    def sample(self, rng: np.random.Generator) -> Any:
        idx = int(rng.integers(0, len(self.choices)))
        return self.choices[idx]


@dataclass
class Condition:
    """
    Simple condition: a parameter is considered active only when
    `expr` evaluates to True given the current config.
    """

    param_name: str
    expr: str  # Python expression using a dict `cfg`


@dataclass
class ParamSpace:
    """
    Defines a hyperparameter space and basic sampling / validation.
    Intended to describe an AutoNSGA-II-like space.
    """

    params: Dict[str, Any] = field(default_factory=dict)
    conditions: List[Condition] = field(default_factory=list)

    def sample(self, rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        """
        Sample a configuration from the space (ignores conditions at sampling time).
        Builder/caller can decide to ignore inactive params.
        """
        rng = np.random.default_rng() if rng is None else rng
        cfg: Dict[str, Any] = {}
        for name, spec in self.params.items():
            if isinstance(spec, (Real, Int, Categorical)):
                cfg[name] = spec.sample(rng)
            else:
                cfg[name] = spec
        return cfg

    def is_active(self, param_name: str, config: Dict[str, Any]) -> bool:
        """
        Returns True if param_name is active given 'config' according to conditions.
        If no condition refers to param_name, it is always active.
        """
        relevant = [c for c in self.conditions if c.param_name == param_name]
        if not relevant:
            return True
        cfg = config
        for cond in relevant:
            active = bool(eval(cond.expr, {}, {"cfg": cfg}))
            if not active:
                return False
        return True

    def validate(self, config: Dict[str, Any]) -> None:
        """
        Basic validation: check all required active params are present and valid.
        """
        for name, spec in self.params.items():
            if not self.is_active(name, config):
                continue
            if name not in config:
                raise ValueError(f"Active parameter '{name}' missing from config")

            value = config[name]
            if isinstance(spec, Real):
                if not (spec.low <= value <= spec.high):
                    raise ValueError(f"Real param '{name}'={value} out of [{spec.low}, {spec.high}]")
            elif isinstance(spec, Int):
                if not (spec.low <= value <= spec.high):
                    raise ValueError(f"Int param '{name}'={value} out of [{spec.low}, {spec.high}]")
            elif isinstance(spec, Categorical):
                if value not in spec.choices:
                    raise ValueError(f"Categorical param '{name}'={value} not in {spec.choices}")


__all__ = ["ParamSpace", "Real", "Int", "Categorical", "Condition"]
