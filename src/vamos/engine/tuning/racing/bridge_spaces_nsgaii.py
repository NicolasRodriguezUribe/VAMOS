"""
NSGA-II configuration space builders for tuning.
"""

from __future__ import annotations

from .bridge_space_parts_discrete import mixed_operator_part, permutation_operator_part_full
from .config_space import AlgorithmConfigSpace, SpacePart, compose_config_space
from .param_space import Boolean, Categorical, Condition, ConditionalBlock, Int, ParamType, Real

# ---------------------------------------------------------------------------
# Core part (shared by ALL NSGA-II encoding variants)
# ---------------------------------------------------------------------------


def _core_part() -> SpacePart:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True),
        Categorical("offspring_ratio", [0.25, 0.5, 0.75, 1.0]),
        Categorical("selection", ["tournament", "boltzmann", "ranking", "sus"]),
        Int("selection_pressure", 2, 10),
        Boolean("use_external_archive"),
        Boolean("archive_unbounded"),
    ]
    archive_type_param = Categorical("archive_type", ["size_cap", "hvc_prune"])
    archive_size_factor_param = Categorical("archive_size_factor", [1, 2, 5, 10])
    conditionals = [
        ConditionalBlock(
            "use_external_archive",
            True,
            [archive_type_param, archive_size_factor_param],
        ),
    ]
    conditions = [
        Condition("archive_type", "cfg['archive_unbounded'] == False"),
        Condition("archive_size_factor", "cfg['archive_unbounded'] == False"),
    ]
    return params, conditionals, conditions


# ---------------------------------------------------------------------------
# Encoding-specific operator parts
# ---------------------------------------------------------------------------


def _real_operator_part() -> SpacePart:
    params: list[ParamType] = [
        Categorical("initializer", ["random", "lhs", "scatter"]),
        Categorical(
            "crossover",
            [
                "sbx", "blx_alpha", "blx_alpha_beta", "arithmetic", "whole_arithmetic",
                "laplace", "fuzzy", "pcx", "undx", "simplex",
            ],
        ),
        Real("crossover_prob", 0.6, 1.0),
        Categorical(
            "mutation",
            [
                "pm", "linked_polynomial", "non_uniform", "gaussian",
                "uniform_reset", "cauchy", "uniform", "levy_flight", "power_law",
            ],
        ),
        Real("mutation_prob_factor", 0.25, 3.0),
        Real("mutation_eta", 5.0, 40.0),
        Categorical("repair", ["none", "clip", "reflect", "random", "round"]),
    ]
    conditionals = [
        ConditionalBlock("crossover", "sbx", [Real("crossover_eta", 5.0, 40.0)]),
        ConditionalBlock(
            "crossover",
            "blx_alpha",
            [
                Real("crossover_alpha", 0.0, 1.0),
                Categorical("blx_repair", ["clip", "random", "reflect", "round"]),
            ],
        ),
        ConditionalBlock(
            "crossover",
            "blx_alpha_beta",
            [Real("blxab_alpha", 0.0, 1.0), Real("blxab_beta", 0.0, 1.0)],
        ),
        ConditionalBlock(
            "crossover",
            "whole_arithmetic",
            [Real("wa_alpha", 0.0, 1.0)],
        ),
        ConditionalBlock(
            "crossover",
            "laplace",
            [Real("laplace_a", -1.0, 1.0), Real("laplace_b", 0.01, 2.0)],
        ),
        ConditionalBlock(
            "crossover",
            "fuzzy",
            [Real("fuzzy_d", 0.0, 2.0)],
        ),
        ConditionalBlock(
            "crossover",
            "pcx",
            [Real("pcx_sigma_eta", 0.01, 0.5), Real("pcx_sigma_zeta", 0.01, 0.5)],
        ),
        ConditionalBlock(
            "crossover",
            "undx",
            [Real("undx_zeta", 0.1, 1.0), Real("undx_eta", 0.1, 1.0)],
        ),
        ConditionalBlock("crossover", "simplex", [Real("simplex_epsilon", 0.1, 1.0)]),
        ConditionalBlock("mutation", "non_uniform", [Real("nonuniform_perturbation", 0.05, 0.5)]),
        ConditionalBlock("mutation", "gaussian", [Real("gaussian_sigma", 0.001, 0.5)]),
        ConditionalBlock("mutation", "cauchy", [Real("cauchy_gamma", 0.001, 0.5)]),
        ConditionalBlock("mutation", "uniform", [Real("uniform_perturb", 0.01, 0.5)]),
        ConditionalBlock(
            "mutation",
            "levy_flight",
            [Real("levy_beta", 0.5, 2.0), Real("levy_scale", 0.001, 0.1)],
        ),
        ConditionalBlock(
            "mutation",
            "power_law",
            [Real("power_index", 0.5, 5.0)],
        ),
        ConditionalBlock(
            "initializer",
            "scatter",
            [Categorical("scatter_base_size_factor", [0.1, 0.2, 0.3, 0.5, 0.75, 1.0])],
        ),
    ]
    return params, conditionals, []


def _permutation_operator_part() -> SpacePart:
    return permutation_operator_part_full(mutation_prob_param="mutation_prob_factor", mutation_prob_bounds=(0.25, 3.0))


def _mixed_operator_part() -> SpacePart:
    return mixed_operator_part(
        crossover_choices=("mixed", "uniform"),
        mutation_choices=("mixed", "gaussian"),
        mutation_prob_param="mutation_prob_factor",
        mutation_prob_bounds=(0.25, 3.0),
    )


def _binary_operator_part() -> SpacePart:
    params: list[ParamType] = [
        Categorical("crossover", ["hux", "uniform", "one_point", "two_point"]),
        Real("crossover_prob", 0.6, 1.0),
        Categorical("mutation", ["bitflip"]),
        Real("mutation_prob_factor", 0.25, 3.0),
    ]
    return params, [], []


def _integer_operator_part() -> SpacePart:
    params: list[ParamType] = [
        Categorical("crossover", ["uniform", "arithmetic", "sbx"]),
        Real("crossover_prob", 0.6, 1.0),
        Categorical("mutation", ["reset", "creep", "pm"]),
        Real("mutation_prob_factor", 0.25, 3.0),
    ]
    conditionals = [
        ConditionalBlock("crossover", "sbx", [Real("crossover_eta", 5.0, 40.0)]),
        ConditionalBlock("mutation", "pm", [Real("mutation_eta", 5.0, 40.0)]),
        ConditionalBlock("mutation", "creep", [Int("creep_step", 1, 5)]),
    ]
    return params, conditionals, []


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def build_nsgaii_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("nsgaii", _core_part(), _real_operator_part())


def build_nsgaii_permutation_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("nsgaii_permutation", _core_part(), _permutation_operator_part())


def build_nsgaii_mixed_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("nsgaii_mixed", _core_part(), _mixed_operator_part())


def build_nsgaii_binary_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("nsgaii_binary", _core_part(), _binary_operator_part())


def build_nsgaii_integer_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("nsgaii_integer", _core_part(), _integer_operator_part())


__all__ = [
    "build_nsgaii_config_space",
    "build_nsgaii_permutation_config_space",
    "build_nsgaii_mixed_config_space",
    "build_nsgaii_binary_config_space",
    "build_nsgaii_integer_config_space",
]
