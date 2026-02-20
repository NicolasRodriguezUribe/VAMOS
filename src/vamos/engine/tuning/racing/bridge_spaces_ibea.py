"""
IBEA configuration space builders for tuning.
"""

from __future__ import annotations

from .bridge_space_parts_discrete import mixed_operator_part, permutation_operator_part_full
from .config_space import AlgorithmConfigSpace, SpacePart, compose_config_space
from .param_space import Categorical, ConditionalBlock, Int, ParamType, Real

# ---------------------------------------------------------------------------
# Core part (shared by ALL IBEA encoding variants)
# ---------------------------------------------------------------------------


def _core_part() -> SpacePart:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True),
        Int("selection_pressure", 2, 10),
        Categorical("indicator", ["eps", "hypervolume"]),
        Real("kappa", 0.01, 0.2),
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
    return mixed_operator_part()


def _binary_operator_part() -> SpacePart:
    params: list[ParamType] = [
        Categorical("crossover", ["hux", "uniform", "one_point", "two_point"]),
        Real("crossover_prob", 0.6, 1.0),
        Categorical("mutation", ["bitflip"]),
        Real("mutation_prob", 0.01, 0.5),
    ]
    return params, [], []


def _integer_operator_part() -> SpacePart:
    params: list[ParamType] = [
        Categorical("crossover", ["uniform", "arithmetic", "sbx"]),
        Real("crossover_prob", 0.6, 1.0),
        Categorical("mutation", ["reset", "creep", "pm"]),
        Real("mutation_prob", 0.01, 0.5),
    ]
    conditionals = [
        ConditionalBlock("crossover", "sbx", [Real("crossover_eta", 5.0, 40.0)]),
        ConditionalBlock("mutation", "pm", [Real("mutation_eta", 5.0, 40.0)]),
        ConditionalBlock("mutation", "creep", [Int("creep_step", 1, 5)]),
    ]
    return params, conditionals, []


def _permutation_operator_part() -> SpacePart:
    return permutation_operator_part_full()


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def build_ibea_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("ibea", _core_part(), _real_operator_part())


def build_ibea_mixed_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("ibea_mixed", _core_part(), _mixed_operator_part())


def build_ibea_binary_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("ibea_binary", _core_part(), _binary_operator_part())


def build_ibea_integer_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("ibea_integer", _core_part(), _integer_operator_part())


def build_ibea_permutation_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("ibea_permutation", _core_part(), _permutation_operator_part())


__all__ = [
    "build_ibea_config_space",
    "build_ibea_mixed_config_space",
    "build_ibea_binary_config_space",
    "build_ibea_integer_config_space",
    "build_ibea_permutation_config_space",
]
