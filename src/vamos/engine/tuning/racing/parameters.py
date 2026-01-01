"""
Re-exports parameter types from param_space for convenience.

All types use signature: ParamType(name, ...)
- Real/FloatParam(name, low, high, log=False)
- Int/IntegerParam(name, low, high, log=False)
- Categorical/CategoricalParam(name, choices)
- Boolean/BooleanParam(name)
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from .param_space import (
    Real,
    Int,
    Categorical,
    Boolean,
    ConditionalBlock,
    ParamType,
    # Aliases
    FloatParam,
    IntegerParam,
    CategoricalParam,
    BooleanParam,
    CategoricalIntegerParam,
)


class BaseParam(Protocol):
    """Protocol for parameter types used in AlgorithmConfigSpace."""

    name: str

    def sample(self, rng: np.random.Generator) -> Any: ...

    def from_unit(self, value: float) -> Any: ...

    def to_unit(self, value: Any) -> float: ...


__all__ = [
    "BaseParam",
    "Real",
    "Int",
    "Categorical",
    "Boolean",
    "ConditionalBlock",
    "ParamType",
    # Aliases
    "FloatParam",
    "IntegerParam",
    "CategoricalParam",
    "BooleanParam",
    "CategoricalIntegerParam",
]
