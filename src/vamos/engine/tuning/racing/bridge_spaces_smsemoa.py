"""
SMS-EMOA configuration space builders for tuning.
"""

from __future__ import annotations

from .config_space import AlgorithmConfigSpace, SpacePart, compose_config_space
from .param_space import Categorical, ConditionalBlock, Int, ParamType, Real

# ---------------------------------------------------------------------------
# Core part (shared by ALL SMS-EMOA encoding variants)
# ---------------------------------------------------------------------------


def _core_part() -> SpacePart:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True),
        Int("selection_pressure", 2, 10),
    ]
    return params, [], []


# ---------------------------------------------------------------------------
# Encoding-specific operator parts
# ---------------------------------------------------------------------------


def _real_operator_part() -> SpacePart:
    params: list[ParamType] = [
        Categorical("crossover", ["sbx"]),
        Real("crossover_prob", 0.6, 1.0),
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


def _binary_operator_part() -> SpacePart:
    params: list[ParamType] = [
        Categorical("crossover", ["one_point", "two_point", "uniform"]),
        Real("crossover_prob", 0.6, 1.0),
        Categorical("mutation", ["bitflip"]),
        Real("mutation_prob", 0.01, 0.5),
    ]
    return params, [], []


def _integer_operator_part() -> SpacePart:
    params: list[ParamType] = [
        Categorical("crossover", ["uniform", "arithmetic"]),
        Real("crossover_prob", 0.6, 1.0),
        Categorical("mutation", ["reset", "creep"]),
        Real("mutation_prob", 0.01, 0.5),
    ]
    conditionals = [
        ConditionalBlock("mutation", "creep", [Int("creep_step", 1, 5)]),
    ]
    return params, conditionals, []


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def build_smsemoa_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("smsemoa", _core_part(), _real_operator_part())


def build_smsemoa_mixed_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("smsemoa_mixed", _core_part(), _mixed_operator_part())


def build_smsemoa_binary_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("smsemoa_binary", _core_part(), _binary_operator_part())


def build_smsemoa_integer_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("smsemoa_integer", _core_part(), _integer_operator_part())


__all__ = [
    "build_smsemoa_config_space",
    "build_smsemoa_mixed_config_space",
    "build_smsemoa_binary_config_space",
    "build_smsemoa_integer_config_space",
]
