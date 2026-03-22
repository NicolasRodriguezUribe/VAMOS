"""Shared typing helpers for algorithm configuration objects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

AlgorithmConfigMapping: TypeAlias = Mapping[str, object]


@runtime_checkable
class AlgorithmConfigProtocol(Protocol):
    def to_dict(self) -> AlgorithmConfigMapping: ...


# ── Operator name literals (for IDE autocompletion) ──────────────────

CrossoverName: TypeAlias = Literal[
    # Real
    "sbx",
    "blx_alpha",
    "blx_alpha_beta",
    "arithmetic",
    "whole_arithmetic",
    "laplace",
    "fuzzy",
    "de",
    "pcx",
    "undx",
    "simplex",
    # Binary
    "one_point",
    "two_point",
    "binary_uniform",
    "hux",
    "spx",
    # Permutation
    "ox",
    "order",
    "pmx",
    "cx",
    "cycle",
    "position_based",
    "erx",
    "aex",
    # Integer
    "int_uniform",
    "int_arithmetic",
    "int_sbx",
    # Mixed
    "mixed",
]

MutationName: TypeAlias = Literal[
    "pm",
    # Real
    "polynomial",
    "gaussian",
    "uniform",
    "uniform_reset",
    "cauchy",
    "non_uniform",
    "linked_polynomial",
    "levy_flight",
    "power_law",
    # Binary
    "bitflip",
    "segment_inversion",
    # Permutation
    "swap",
    "insert",
    "scramble",
    "inversion",
    "displacement",
    "two_opt",
    # Integer
    "reset",
    "int_pm",
    "creep",
    "boundary",
    "int_gaussian",
    # Mixed
    "mixed_mutation",
]

ProbabilityExpression: TypeAlias = Literal["1/n"]
ProbabilityValue: TypeAlias = float | ProbabilityExpression

SelectionName: TypeAlias = Literal["tournament"]

InitializerName: TypeAlias = Literal[
    "random",
    "lhs",
    "scatter",
    "scatter_search",
    "sobol",
    "halton",
    "obl",
    "opposition",
    "custom",
]

RepairName: TypeAlias = Literal[
    "clip",
    "clamp",
    "reflect",
    "random",
    "resample",
    "round",
    "wrap",
    "wrapping",
    "midpoint",
    "midpoint_base",
    "gradient",
]
RepairConfigValue: TypeAlias = tuple[RepairName, dict[str, Any]] | Literal["auto"]

# ── Algorithm, engine, and aggregation literals ──────────────────────

AlgorithmName: TypeAlias = Literal[
    "auto",
    "nsgaii",
    "nsgaiii",
    "moead",
    "smsemoa",
    "spea2",
    "ibea",
    "smpso",
    "agemoea",
    "rvea",
]

EngineName: TypeAlias = Literal["numpy", "numba", "moocore", "jax"]

AggregationName: TypeAlias = Literal[
    "pbi",
    "tchebycheff",
    "weighted_sum",
    "modified_tchebycheff",
]

__all__ = [
    "AlgorithmConfigMapping",
    "AlgorithmConfigProtocol",
    "CrossoverName",
    "MutationName",
    "ProbabilityExpression",
    "ProbabilityValue",
    "SelectionName",
    "InitializerName",
    "RepairName",
    "RepairConfigValue",
    "AlgorithmName",
    "EngineName",
    "AggregationName",
]
