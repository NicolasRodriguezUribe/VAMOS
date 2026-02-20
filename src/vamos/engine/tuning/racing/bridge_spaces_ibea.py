"""
IBEA configuration space builders for tuning.
"""

from __future__ import annotations

from .bridge_space_parts_discrete import (
    binary_operator_part_full,
    integer_operator_part_full,
    mixed_operator_part,
    permutation_operator_part_full,
    real_operator_part_medium,
)
from .config_space import AlgorithmConfigSpace, SpacePart, compose_config_space
from .param_space import Categorical, Int, ParamType, Real

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
    return real_operator_part_medium()


def _mixed_operator_part() -> SpacePart:
    return mixed_operator_part(
        crossover_choices=("mixed", "uniform"),
        mutation_choices=("mixed", "gaussian"),
    )


def _binary_operator_part() -> SpacePart:
    return binary_operator_part_full()


def _integer_operator_part() -> SpacePart:
    return integer_operator_part_full()


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
