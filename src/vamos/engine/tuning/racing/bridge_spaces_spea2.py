"""
SPEA2 configuration space builders for tuning.
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
from .param_space import Int, ParamType

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
    return real_operator_part_medium()


def _mixed_operator_part() -> SpacePart:
    return mixed_operator_part(
        crossover_choices=("mixed", "uniform"),
        mutation_choices=("mixed", "gaussian"),
    )


def _permutation_operator_part() -> SpacePart:
    return permutation_operator_part_full()


def _binary_operator_part() -> SpacePart:
    return binary_operator_part_full()


def _integer_operator_part() -> SpacePart:
    return integer_operator_part_full()


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def build_spea2_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("spea2", _core_part(), _real_operator_part())


def build_spea2_mixed_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("spea2_mixed", _core_part(), _mixed_operator_part())


def build_spea2_permutation_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("spea2_permutation", _core_part(), _permutation_operator_part())


def build_spea2_binary_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("spea2_binary", _core_part(), _binary_operator_part())


def build_spea2_integer_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("spea2_integer", _core_part(), _integer_operator_part())


__all__ = [
    "build_spea2_config_space",
    "build_spea2_mixed_config_space",
    "build_spea2_permutation_config_space",
    "build_spea2_binary_config_space",
    "build_spea2_integer_config_space",
]
