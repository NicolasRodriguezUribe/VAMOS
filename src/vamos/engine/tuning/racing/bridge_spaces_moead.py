"""
MOEA-D configuration space builders for tuning.
"""

from __future__ import annotations

from .config_space import AlgorithmConfigSpace
from .param_space import Boolean, Categorical, ConditionalBlock, Int, ParamType, Real


def build_moead_config_space() -> AlgorithmConfigSpace:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True),
        Int("neighbor_size", 5, 40),
        Real("delta", 0.5, 0.95),
        Int("replace_limit", 1, 5),
        Categorical("crossover", ["sbx", "de"]),
        Categorical("mutation", ["pm"]),
        Real("mutation_prob", 0.01, 0.5),
        Real("mutation_eta", 5.0, 40.0),
        Categorical("aggregation", ["tchebycheff", "weighted_sum", "pbi"]),
    ]
    conditionals = [
        ConditionalBlock("crossover", "sbx", [Real("crossover_prob", 0.6, 1.0), Real("crossover_eta", 10.0, 40.0)]),
        ConditionalBlock("crossover", "de", [Real("de_cr", 0.0, 1.0), Real("de_f", 0.0, 1.0)]),
        ConditionalBlock("aggregation", "pbi", [Real("pbi_theta", 1.0, 10.0)]),
    ]
    return AlgorithmConfigSpace("moead", params, conditionals)


def build_moead_permutation_config_space() -> AlgorithmConfigSpace:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True),
        Int("neighbor_size", 5, 40),
        Real("delta", 0.5, 0.95),
        Int("replace_limit", 1, 5),
        Categorical("crossover", ["ox", "pmx", "edge", "cycle", "position"]),
        Real("crossover_prob", 0.6, 1.0),
        Categorical("mutation", ["swap", "insert", "scramble", "inversion", "displacement"]),
        Real("mutation_prob", 0.01, 0.5),
        Categorical("aggregation", ["tchebycheff", "weighted_sum", "pbi", "modified_tchebycheff"]),
        Boolean("use_external_archive"),
    ]
    archive_type_param = Categorical("archive_type", ["hypervolume", "crowding"])
    archive_size_factor_param = Categorical("archive_size_factor", [1, 2, 5, 10])
    conditionals = [
        ConditionalBlock("aggregation", "pbi", [Real("pbi_theta", 1.0, 10.0)]),
        ConditionalBlock("aggregation", "modified_tchebycheff", [Real("mtch_rho", 0.0001, 0.1)]),
        ConditionalBlock(
            "use_external_archive",
            True,
            [
                archive_type_param,
                archive_size_factor_param,
            ],
        ),
    ]
    return AlgorithmConfigSpace("moead_permutation", params, conditionals)


def build_moead_binary_config_space() -> AlgorithmConfigSpace:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True),
        Int("neighbor_size", 5, 40),
        Real("delta", 0.5, 0.95),
        Int("replace_limit", 1, 5),
        Categorical("crossover", ["one_point", "two_point", "uniform"]),
        Real("crossover_prob", 0.6, 1.0),
        Categorical("mutation", ["bitflip"]),
        Real("mutation_prob", 0.01, 0.5),
        Categorical("aggregation", ["tchebycheff", "weighted_sum", "pbi"]),
    ]
    conditionals = [
        ConditionalBlock("aggregation", "pbi", [Real("pbi_theta", 1.0, 10.0)]),
    ]
    return AlgorithmConfigSpace("moead_binary", params, conditionals)


def build_moead_integer_config_space() -> AlgorithmConfigSpace:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True),
        Int("neighbor_size", 5, 40),
        Real("delta", 0.5, 0.95),
        Int("replace_limit", 1, 5),
        Categorical("crossover", ["uniform", "arithmetic", "sbx"]),
        Real("crossover_prob", 0.6, 1.0),
        Categorical("mutation", ["reset", "creep", "pm"]),
        Real("mutation_prob", 0.01, 0.5),
        Categorical("aggregation", ["tchebycheff", "weighted_sum", "pbi"]),
    ]
    conditionals = [
        ConditionalBlock("crossover", "sbx", [Real("crossover_eta", 5.0, 40.0)]),
        ConditionalBlock("mutation", "pm", [Real("mutation_eta", 5.0, 40.0)]),
        ConditionalBlock("mutation", "creep", [Int("creep_step", 1, 5)]),
        ConditionalBlock("aggregation", "pbi", [Real("pbi_theta", 1.0, 10.0)]),
    ]
    return AlgorithmConfigSpace("moead_integer", params, conditionals)


__all__ = [
    "build_moead_config_space",
    "build_moead_permutation_config_space",
    "build_moead_binary_config_space",
    "build_moead_integer_config_space",
]
