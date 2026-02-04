"""
Configuration space builders for tuning.
"""

from __future__ import annotations

from .config_space import AlgorithmConfigSpace
from .param_space import Boolean, Categorical, Condition, ConditionalBlock, Int, ParamType, Real


def build_nsgaii_config_space() -> AlgorithmConfigSpace:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True),
        Categorical(
            "offspring_ratio",
            [0.25, 0.5, 0.75, 1.0],
        ),
        Categorical("initializer", ["random", "lhs", "scatter"]),
        Categorical("crossover", ["sbx", "blx_alpha", "arithmetic", "pcx", "undx", "simplex"]),
        Real("crossover_prob", 0.6, 1.0),
        Categorical("mutation", ["pm", "linked_polynomial", "non_uniform", "gaussian", "uniform_reset", "cauchy", "uniform"]),
        Real("mutation_prob_factor", 0.25, 3.0),
        Real("mutation_eta", 5.0, 40.0),
        Categorical("selection", ["tournament"]),
        Int("selection_pressure", 2, 10),
        Categorical("repair", ["none", "clip", "reflect", "random", "round"]),
        Boolean("use_external_archive"),
        Boolean("archive_unbounded"),
    ]
    archive_type_param = Categorical("archive_type", ["hypervolume", "crowding"])
    archive_size_factor_param = Categorical("archive_size_factor", [1, 2, 5, 10])
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
            "pcx",
            [
                Real("pcx_sigma_eta", 0.01, 0.5),
                Real("pcx_sigma_zeta", 0.01, 0.5),
            ],
        ),
        ConditionalBlock(
            "crossover",
            "undx",
            [
                Real("undx_zeta", 0.1, 1.0),
                Real("undx_eta", 0.1, 1.0),
            ],
        ),
        ConditionalBlock("crossover", "simplex", [Real("simplex_epsilon", 0.1, 1.0)]),
        ConditionalBlock("mutation", "non_uniform", [Real("nonuniform_perturbation", 0.05, 0.5)]),
        ConditionalBlock("mutation", "gaussian", [Real("gaussian_sigma", 0.001, 0.5)]),
        ConditionalBlock("mutation", "cauchy", [Real("cauchy_gamma", 0.001, 0.5)]),
        ConditionalBlock("mutation", "uniform", [Real("uniform_perturb", 0.01, 0.5)]),
        ConditionalBlock(
            "initializer",
            "scatter",
            [
                Categorical("scatter_base_size_factor", [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]),
            ],
        ),
        ConditionalBlock(
            "use_external_archive",
            True,
            [
                archive_type_param,
                archive_size_factor_param,
            ],
        ),
    ]
    conditions = [
        Condition("archive_type", "cfg['archive_unbounded'] == False"),
        Condition("archive_size_factor", "cfg['archive_unbounded'] == False"),
    ]
    return AlgorithmConfigSpace("nsgaii", params, conditionals, conditions)


def build_nsgaii_permutation_config_space() -> AlgorithmConfigSpace:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True),
        Categorical(
            "offspring_ratio",
            [0.25, 0.5, 0.75, 1.0],
        ),
        Categorical("crossover", ["ox", "pmx", "edge", "cycle", "position"]),
        Real("crossover_prob", 0.6, 1.0),
        Categorical("mutation", ["swap", "insert", "scramble", "inversion", "displacement"]),
        Real("mutation_prob_factor", 0.25, 3.0),
        Categorical("selection", ["tournament"]),
        Int("selection_pressure", 2, 10),
        Boolean("use_external_archive"),
        Boolean("archive_unbounded"),
    ]
    archive_type_param = Categorical("archive_type", ["hypervolume", "crowding"])
    archive_size_factor_param = Categorical("archive_size_factor", [1, 2, 5, 10])
    conditionals = [
        ConditionalBlock(
            "use_external_archive",
            True,
            [
                archive_type_param,
                archive_size_factor_param,
            ],
        ),
    ]
    conditions = [
        Condition("archive_type", "cfg['archive_unbounded'] == False"),
        Condition("archive_size_factor", "cfg['archive_unbounded'] == False"),
    ]
    return AlgorithmConfigSpace("nsgaii_permutation", params, conditionals, conditions)


def build_nsgaii_mixed_config_space() -> AlgorithmConfigSpace:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True),
        Categorical(
            "offspring_ratio",
            [0.25, 0.5, 0.75, 1.0],
        ),
        Categorical("crossover", ["mixed", "uniform"]),
        Real("crossover_prob", 0.6, 1.0),
        Categorical("mutation", ["mixed", "gaussian"]),
        Real("mutation_prob_factor", 0.25, 3.0),
        Categorical("selection", ["tournament"]),
        Int("selection_pressure", 2, 10),
        Boolean("use_external_archive"),
        Boolean("archive_unbounded"),
    ]
    archive_type_param = Categorical("archive_type", ["hypervolume", "crowding"])
    archive_size_factor_param = Categorical("archive_size_factor", [1, 2, 5, 10])
    conditionals = [
        ConditionalBlock(
            "use_external_archive",
            True,
            [
                archive_type_param,
                archive_size_factor_param,
            ],
        ),
    ]
    conditions = [
        Condition("archive_type", "cfg['archive_unbounded'] == False"),
        Condition("archive_size_factor", "cfg['archive_unbounded'] == False"),
    ]
    return AlgorithmConfigSpace("nsgaii_mixed", params, conditionals, conditions)


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


def build_nsgaiii_config_space() -> AlgorithmConfigSpace:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True),
        Categorical("crossover", ["sbx"]),
        Real("crossover_prob", 0.6, 1.0),
        Real("crossover_eta", 20.0, 40.0),
        Categorical("mutation", ["pm"]),
        Real("mutation_prob", 0.01, 0.5),
        Real("mutation_eta", 5.0, 40.0),
        Int("selection_pressure", 2, 10),
    ]
    return AlgorithmConfigSpace("nsgaiii", params, [])


def build_smsemoa_config_space() -> AlgorithmConfigSpace:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True),
        Categorical("crossover", ["sbx"]),
        Real("crossover_prob", 0.6, 1.0),
        Real("crossover_eta", 10.0, 40.0),
        Categorical("mutation", ["pm"]),
        Real("mutation_prob", 0.01, 0.5),
        Real("mutation_eta", 5.0, 40.0),
        Int("selection_pressure", 2, 10),
    ]
    return AlgorithmConfigSpace("smsemoa", params, [])


def build_spea2_config_space() -> AlgorithmConfigSpace:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True),
        Int("archive_size", 20, 200, log=True),
        Categorical("crossover", ["sbx"]),
        Real("crossover_prob", 0.6, 0.95),
        Real("crossover_eta", 10.0, 40.0),
        Categorical("mutation", ["pm"]),
        Real("mutation_prob", 0.01, 0.5),
        Real("mutation_eta", 5.0, 40.0),
        Int("selection_pressure", 2, 10),
        Int("k_neighbors", 1, 25),
    ]
    return AlgorithmConfigSpace("spea2", params, [])


def build_ibea_config_space() -> AlgorithmConfigSpace:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True),
        Categorical("crossover", ["sbx"]),
        Real("crossover_prob", 0.6, 0.95),
        Real("crossover_eta", 10.0, 40.0),
        Categorical("mutation", ["pm"]),
        Real("mutation_prob", 0.01, 0.5),
        Real("mutation_eta", 5.0, 40.0),
        Int("selection_pressure", 2, 10),
        Categorical("indicator", ["eps", "hypervolume"]),
        Real("kappa", 0.01, 0.2),
    ]
    return AlgorithmConfigSpace("ibea", params, [])


def build_smpso_config_space() -> AlgorithmConfigSpace:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True),
        Int("archive_size", 20, 200, log=True),
        Real("inertia", 0.1, 0.9),
        Real("c1", 0.5, 2.5),
        Real("c2", 0.5, 2.5),
        Real("vmax_fraction", 0.1, 1.0),
        Real("mutation_prob", 0.01, 0.5),
        Real("mutation_eta", 5.0, 40.0),
    ]
    return AlgorithmConfigSpace("smpso", params, [])


__all__ = [
    "build_nsgaii_config_space",
    "build_nsgaii_permutation_config_space",
    "build_nsgaii_mixed_config_space",
    "build_moead_config_space",
    "build_moead_permutation_config_space",
    "build_nsgaiii_config_space",
    "build_smsemoa_config_space",
    "build_spea2_config_space",
    "build_ibea_config_space",
    "build_smpso_config_space",
]
