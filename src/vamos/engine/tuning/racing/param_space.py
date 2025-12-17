"""
Hyperparameter space definitions for VAMOS tuning.

All parameter types use name as the first argument:
- Real(name, low, high, log=False)
- Int(name, low, high, log=False)
- Categorical(name, choices)
- Boolean(name)

All types support:
- sample(rng) - draw a random value
- to_unit(value) - map to [0, 1] space for optimization
- from_unit(value) - map from [0, 1] space back to parameter value
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np


@dataclass
class Real:
    """Real-valued hyperparameter in [low, high]."""

    name: str
    low: float
    high: float
    log: bool = False

    def sample(self, rng: np.random.Generator) -> float:
        if self.log:
            lo, hi = math.log(self.low), math.log(self.high)
            return float(math.exp(rng.uniform(lo, hi)))
        return float(rng.uniform(self.low, self.high))

    def to_unit(self, value: float) -> float:
        """Map value to [0, 1] space."""
        v = float(value)
        if self.log:
            lo, hi = math.log(self.low), math.log(self.high)
            return 0.0 if hi == lo else (math.log(v) - lo) / (hi - lo)
        return 0.0 if self.high == self.low else (v - self.low) / (self.high - self.low)

    def from_unit(self, value: float) -> float:
        """Map from [0, 1] space to parameter value."""
        u = min(max(value, 0.0), 1.0)
        if self.log:
            lo, hi = math.log(self.low), math.log(self.high)
            return float(math.exp(lo + u * (hi - lo)))
        return float(self.low + u * (self.high - self.low))


@dataclass
class Int:
    """Integer hyperparameter in [low, high] (inclusive)."""

    name: str
    low: int
    high: int
    log: bool = False

    def sample(self, rng: np.random.Generator) -> int:
        if self.log:
            lo, hi = math.log(self.low), math.log(self.high)
            return int(round(math.exp(rng.uniform(lo, hi))))
        return int(rng.integers(self.low, self.high + 1))

    def to_unit(self, value: int) -> float:
        """Map value to [0, 1] space."""
        v = float(value)
        if self.log:
            lo, hi = math.log(self.low), math.log(self.high)
            return 0.0 if hi == lo else (math.log(v) - lo) / (hi - lo)
        return 0.0 if self.high == self.low else (v - self.low) / (self.high - self.low)

    def from_unit(self, value: float) -> int:
        """Map from [0, 1] space to parameter value."""
        u = min(max(value, 0.0), 1.0)
        if self.log:
            lo, hi = math.log(self.low), math.log(self.high)
            mapped = math.exp(lo + u * (hi - lo))
        else:
            mapped = self.low + u * (self.high - self.low)
        return int(round(mapped))


@dataclass
class Categorical:
    """Categorical hyperparameter with discrete choices."""

    name: str
    choices: Sequence[Any]

    def sample(self, rng: np.random.Generator) -> Any:
        return self.choices[int(rng.integers(0, len(self.choices)))]

    def to_unit(self, value: Any) -> float:
        """Map value to [0, 1] space based on choice index."""
        idx = list(self.choices).index(value)
        return 0.0 if len(self.choices) == 1 else idx / float(len(self.choices) - 1)

    def from_unit(self, value: float) -> Any:
        """Map from [0, 1] space to a choice."""
        u = min(max(value, 0.0), 1.0)
        idx = int(round(u * (len(self.choices) - 1)))
        return self.choices[idx]


@dataclass
class Boolean:
    """Boolean hyperparameter (True/False)."""

    name: str

    def sample(self, rng: np.random.Generator) -> bool:
        return bool(rng.integers(0, 2))

    def to_unit(self, value: bool) -> float:
        return 1.0 if value else 0.0

    def from_unit(self, value: float) -> bool:
        return value >= 0.5


@dataclass
class Condition:
    """
    Simple condition: a parameter is considered active only when
    `expr` evaluates to True given the current config.
    """

    param_name: str
    expr: str  # Python expression using a dict `cfg`


@dataclass
class ConditionalBlock:
    """
    A block of parameters that are active only when a parent parameter
    has a specific value.
    """

    parent_name: str
    parent_value: Any
    params: List[Any]  # List of Real, Int, Categorical, etc.


# Type alias for any parameter type
ParamType = Union[Real, Int, Categorical, Boolean]


@dataclass
class ParamSpace:
    """
    Defines a hyperparameter space with named parameters.

    Example:
        space = ParamSpace(params={
            "lr": Real("lr", 0.001, 0.1, log=True),
            "epochs": Int("epochs", 10, 100),
        })
    """

    params: Dict[str, ParamType] = field(default_factory=dict)
    conditions: List[Condition] = field(default_factory=list)

    def sample(self, rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        """Sample a configuration from the space."""
        rng = np.random.default_rng() if rng is None else rng
        return {name: spec.sample(rng) for name, spec in self.params.items()}

    def is_active(self, param_name: str, config: Dict[str, Any]) -> bool:
        """Check if param is active given config and conditions."""
        relevant = [c for c in self.conditions if c.param_name == param_name]
        if not relevant:
            return True
        cfg = config
        return all(bool(eval(c.expr, {}, {"cfg": cfg})) for c in relevant)

    def validate(self, config: Dict[str, Any]) -> None:
        """Validate that all active params are present and within bounds."""
        for name, spec in self.params.items():
            if not self.is_active(name, config):
                continue
            if name not in config:
                raise ValueError(f"Active parameter '{name}' missing from config")

            value = config[name]
            if isinstance(spec, Real):
                if not (spec.low <= value <= spec.high):
                    raise ValueError(
                        f"Real param '{name}'={value} out of [{spec.low}, {spec.high}]"
                    )
            elif isinstance(spec, Int):
                if not (spec.low <= value <= spec.high):
                    raise ValueError(
                        f"Int param '{name}'={value} out of [{spec.low}, {spec.high}]"
                    )
            elif isinstance(spec, Categorical):
                if value not in spec.choices:
                    raise ValueError(
                        f"Categorical param '{name}'={value} not in {spec.choices}"
                    )


# Aliases
FloatParam = Real
IntegerParam = Int
CategoricalParam = Categorical
BooleanParam = Boolean
CategoricalIntegerParam = Categorical


__all__ = [
    "ParamSpace",
    "Real",
    "Int",
    "Categorical",
    "Boolean",
    "Condition",
    "ConditionalBlock",
    "ParamType",
    # Aliases
    "FloatParam",
    "IntegerParam",
    "CategoricalParam",
    "BooleanParam",
    "CategoricalIntegerParam",
]
