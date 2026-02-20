"""
RVEA configuration space builders for tuning.
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
# Core part (shared by ALL RVEA encoding variants)
# ---------------------------------------------------------------------------


def _core_part() -> SpacePart:
    params: list[ParamType] = [
        Int("n_partitions", 4, 12),
        Real("alpha", 1.0, 4.0),
        Real("adapt_freq", 0.05, 0.3),
        Boolean("use_external_archive"),
    ]
    archive_type_param = Categorical("archive_type", ["size_cap", "epsilon_grid", "hvc_prune", "hybrid"])
    archive_size_factor_param = Categorical("archive_size_factor", [1, 2, 5, 10])
    archive_prune_policy_param = Categorical(
        "archive_prune_policy", ["crowding", "hv_contrib", "mc_hv_contrib", "random"]
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


def build_rvea_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("rvea", _core_part(), _real_operator_part())


def build_rvea_mixed_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("rvea_mixed", _core_part(), _mixed_operator_part())


def build_rvea_permutation_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("rvea_permutation", _core_part(), _permutation_operator_part())


def build_rvea_binary_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("rvea_binary", _core_part(), _binary_operator_part())


def build_rvea_integer_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("rvea_integer", _core_part(), _integer_operator_part())


__all__ = [
    "build_rvea_config_space",
    "build_rvea_mixed_config_space",
    "build_rvea_permutation_config_space",
    "build_rvea_binary_config_space",
    "build_rvea_integer_config_space",
]
