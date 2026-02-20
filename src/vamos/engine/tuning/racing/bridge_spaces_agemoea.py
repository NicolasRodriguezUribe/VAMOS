"""
AGE-MOEA configuration space builders for tuning.
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
from .param_space import Boolean, Categorical, ConditionalBlock, Int, ParamType, Real

# ---------------------------------------------------------------------------
# Core part (shared by ALL AGE-MOEA encoding variants)
# ---------------------------------------------------------------------------


def _core_part() -> SpacePart:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True),
        Boolean("use_external_archive"),
    ]
    archive_type_param = Categorical("archive_type", ["size_cap", "epsilon_grid", "hvc_prune", "hybrid"])
    archive_size_factor_param = Categorical("archive_size_factor", [1, 2, 5, 10])
    archive_prune_policy_param = Categorical(
        "archive_prune_policy", ["crowding", "hv_contrib", "mc_hv_contrib", "spea2", "random"]
    )
    archive_epsilon_param = Real("archive_epsilon", 1e-4, 0.1, log=True)
    conditionals = [
        ConditionalBlock(
            "use_external_archive",
            True,
            [archive_type_param, archive_size_factor_param, archive_prune_policy_param, archive_epsilon_param],
        ),
    ]
    return params, conditionals, []


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


def build_agemoea_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("agemoea", _core_part(), _real_operator_part())


def build_agemoea_mixed_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("agemoea_mixed", _core_part(), _mixed_operator_part())


def build_agemoea_permutation_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("agemoea_permutation", _core_part(), _permutation_operator_part())


def build_agemoea_binary_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("agemoea_binary", _core_part(), _binary_operator_part())


def build_agemoea_integer_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("agemoea_integer", _core_part(), _integer_operator_part())


__all__ = [
    "build_agemoea_config_space",
    "build_agemoea_mixed_config_space",
    "build_agemoea_permutation_config_space",
    "build_agemoea_binary_config_space",
    "build_agemoea_integer_config_space",
]
