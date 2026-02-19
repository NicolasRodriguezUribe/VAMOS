"""
MOEA/D configuration space builders for tuning.
"""

from __future__ import annotations

from .config_space import AlgorithmConfigSpace, SpacePart, compose_config_space
from .param_space import Boolean, Categorical, ConditionalBlock, Int, ParamType, Real

# ---------------------------------------------------------------------------
# Core part (shared by ALL MOEA/D encoding variants)
# ---------------------------------------------------------------------------


def _core_part() -> SpacePart:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True),
        Int("neighbor_size", 5, 40),
        Real("delta", 0.5, 0.95),
        Int("replace_limit", 1, 5),
    ]
    return params, [], []


def _aggregation_part(*, extended: bool = False) -> SpacePart:
    """Aggregation params.  *extended* adds ``modified_tchebycheff`` (permutation variant)."""
    choices = ["tchebycheff", "weighted_sum", "pbi"]
    if extended:
        choices.append("modified_tchebycheff")
    conditionals: list[ConditionalBlock] = [
        ConditionalBlock("aggregation", "pbi", [Real("pbi_theta", 1.0, 10.0)]),
    ]
    if extended:
        conditionals.append(
            ConditionalBlock("aggregation", "modified_tchebycheff", [Real("mtch_rho", 0.0001, 0.1)]),
        )
    return [Categorical("aggregation", choices)], conditionals, []


# ---------------------------------------------------------------------------
# Encoding-specific operator parts
# ---------------------------------------------------------------------------


def _real_operator_part() -> SpacePart:
    params: list[ParamType] = [
        Categorical("crossover", ["sbx", "de"]),
        Categorical("mutation", ["pm"]),
        Real("mutation_prob", 0.01, 0.5),
        Real("mutation_eta", 5.0, 40.0),
    ]
    conditionals = [
        ConditionalBlock("crossover", "sbx", [Real("crossover_prob", 0.6, 1.0), Real("crossover_eta", 10.0, 40.0)]),
        ConditionalBlock("crossover", "de", [Real("de_cr", 0.0, 1.0), Real("de_f", 0.0, 1.0)]),
    ]
    return params, conditionals, []


def _permutation_operator_part() -> SpacePart:
    params: list[ParamType] = [
        Categorical("crossover", ["ox", "pmx", "edge", "cycle", "position"]),
        Real("crossover_prob", 0.6, 1.0),
        Categorical("mutation", ["swap", "insert", "scramble", "inversion", "displacement"]),
        Real("mutation_prob", 0.01, 0.5),
        Boolean("use_external_archive"),
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
    return params, conditionals, []


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


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def build_moead_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("moead", _core_part(), _aggregation_part(), _real_operator_part())


def build_moead_permutation_config_space() -> AlgorithmConfigSpace:
    return compose_config_space(
        "moead_permutation",
        _core_part(),
        _aggregation_part(extended=True),
        _permutation_operator_part(),
    )


def build_moead_mixed_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("moead_mixed", _core_part(), _aggregation_part(), _mixed_operator_part())


def build_moead_binary_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("moead_binary", _core_part(), _aggregation_part(), _binary_operator_part())


def build_moead_integer_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("moead_integer", _core_part(), _aggregation_part(), _integer_operator_part())


__all__ = [
    "build_moead_config_space",
    "build_moead_permutation_config_space",
    "build_moead_mixed_config_space",
    "build_moead_binary_config_space",
    "build_moead_integer_config_space",
]
