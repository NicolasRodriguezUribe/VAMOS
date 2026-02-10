"""
SPEA2 configuration space builders for tuning.
"""

from __future__ import annotations

from .config_space import AlgorithmConfigSpace, SpacePart, compose_config_space
from .param_space import Categorical, Int, ParamType, Real

# ---------------------------------------------------------------------------
# Core part (shared by ALL SPEA2 encoding variants)
# ---------------------------------------------------------------------------


def _core_part() -> SpacePart:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True),
        Int("archive_size", 20, 200, log=True),
        Int("selection_pressure", 2, 10),
        Int("k_neighbors", 1, 25),
    ]
    return params, [], []


# ---------------------------------------------------------------------------
# Encoding-specific operator parts
# ---------------------------------------------------------------------------


def _real_operator_part() -> SpacePart:
    params: list[ParamType] = [
        Categorical("crossover", ["sbx"]),
        Real("crossover_prob", 0.6, 0.95),
        Real("crossover_eta", 10.0, 40.0),
        Categorical("mutation", ["pm"]),
        Real("mutation_prob", 0.01, 0.5),
        Real("mutation_eta", 5.0, 40.0),
    ]
    return params, [], []


def _mixed_operator_part() -> SpacePart:
    params: list[ParamType] = [
        Categorical("crossover", ["mixed"]),
        Real("crossover_prob", 0.6, 1.0),
        Categorical("mutation", ["mixed"]),
        Real("mutation_prob", 0.01, 0.5),
    ]
    return params, [], []


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def build_spea2_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("spea2", _core_part(), _real_operator_part())


def build_spea2_mixed_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("spea2_mixed", _core_part(), _mixed_operator_part())


__all__ = [
    "build_spea2_config_space",
    "build_spea2_mixed_config_space",
]
