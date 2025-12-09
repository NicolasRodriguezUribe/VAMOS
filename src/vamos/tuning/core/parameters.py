"""
Typed hyperparameter primitives for algorithm configuration spaces.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np


class BaseParam:
    name: str

    def sample(self, rng: np.random.Generator) -> Any:
        raise NotImplementedError

    def from_unit(self, value: float) -> Any:
        raise NotImplementedError

    def to_unit(self, value: Any) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class CategoricalParam(BaseParam):
    name: str
    choices: List[Any]

    def sample(self, rng: np.random.Generator) -> Any:
        idx = int(rng.integers(0, len(self.choices)))
        return self.choices[idx]

    def from_unit(self, value: float) -> Any:
        clipped = min(max(value, 0.0), 1.0)
        idx = int(round(clipped * (len(self.choices) - 1)))
        return self.choices[idx]

    def to_unit(self, value: Any) -> float:
        idx = self.choices.index(value)
        if len(self.choices) == 1:
            return 0.0
        return idx / float(len(self.choices) - 1)


@dataclass(frozen=True)
class CategoricalIntegerParam(BaseParam):
    name: str
    choices: List[int]

    def sample(self, rng: np.random.Generator) -> int:
        idx = int(rng.integers(0, len(self.choices)))
        return int(self.choices[idx])

    def from_unit(self, value: float) -> int:
        clipped = min(max(value, 0.0), 1.0)
        idx = int(round(clipped * (len(self.choices) - 1)))
        return int(self.choices[idx])

    def to_unit(self, value: int) -> float:
        idx = self.choices.index(int(value))
        if len(self.choices) == 1:
            return 0.0
        return idx / float(len(self.choices) - 1)


@dataclass(frozen=True)
class IntegerParam(BaseParam):
    name: str
    low: int
    high: int
    log: bool = False

    def sample(self, rng: np.random.Generator) -> int:
        if self.log:
            u = rng.random()
            return int(self.from_unit(u))
        return int(rng.integers(self.low, self.high + 1))

    def from_unit(self, value: float) -> int:
        u = min(max(value, 0.0), 1.0)
        if self.log:
            lo, hi = np.log(self.low), np.log(self.high)
            mapped = np.exp(lo + u * (hi - lo))
        else:
            mapped = self.low + u * (self.high - self.low)
        return int(round(mapped))

    def to_unit(self, value: int) -> float:
        v = float(value)
        if self.log:
            lo, hi = np.log(self.low), np.log(self.high)
            return (np.log(v) - lo) / (hi - lo)
        return (v - self.low) / (self.high - self.low)


@dataclass(frozen=True)
class FloatParam(BaseParam):
    name: str
    low: float
    high: float
    log: bool = False

    def sample(self, rng: np.random.Generator) -> float:
        u = float(rng.random())
        return float(self.from_unit(u))

    def from_unit(self, value: float) -> float:
        u = min(max(value, 0.0), 1.0)
        if self.log:
            lo, hi = np.log(self.low), np.log(self.high)
            return float(np.exp(lo + u * (hi - lo)))
        return float(self.low + u * (self.high - self.low))

    def to_unit(self, value: float) -> float:
        v = float(value)
        if self.log:
            lo, hi = np.log(self.low), np.log(self.high)
            return (np.log(v) - lo) / (hi - lo)
        return (v - self.low) / (self.high - self.low)


@dataclass(frozen=True)
class BooleanParam(BaseParam):
    name: str

    def sample(self, rng: np.random.Generator) -> bool:
        return bool(rng.integers(0, 2))

    def from_unit(self, value: float) -> bool:
        return value >= 0.5

    def to_unit(self, value: bool) -> float:
        return 1.0 if value else 0.0


@dataclass(frozen=True)
class ConditionalBlock:
    parent_name: str
    parent_value: Any
    params: List[BaseParam]
