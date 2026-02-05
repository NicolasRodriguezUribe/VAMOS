"""
RVEA configuration space builders for tuning.
"""

from __future__ import annotations

from .config_space import AlgorithmConfigSpace
from .param_space import Boolean, Categorical, ConditionalBlock, Int, ParamType, Real


def build_rvea_config_space() -> AlgorithmConfigSpace:
    params: list[ParamType] = [
        Int("n_partitions", 4, 12),
        Real("alpha", 1.0, 4.0),
        Real("adapt_freq", 0.05, 0.3),
        Categorical("initializer", ["random", "lhs", "scatter"]),
        Categorical("crossover", ["sbx", "blx_alpha", "arithmetic", "pcx", "undx", "simplex"]),
        Real("crossover_prob", 0.6, 1.0),
        Categorical(
            "mutation",
            ["pm", "linked_polynomial", "non_uniform", "gaussian", "uniform_reset", "cauchy", "uniform"],
        ),
        Real("mutation_prob", 0.01, 0.5),
        Real("mutation_eta", 5.0, 40.0),
        Categorical("repair", ["none", "clip", "reflect", "random", "round"]),
        Boolean("use_external_archive"),
    ]
    archive_type_param = Categorical("archive_type", ["size_cap", "epsilon_grid", "hvc_prune", "hybrid"])
    archive_size_factor_param = Categorical("archive_size_factor", [1, 2, 5, 10])
    archive_prune_policy_param = Categorical("archive_prune_policy", ["crowding", "hv_contrib", "mc_hv_contrib", "random"])
    archive_epsilon_param = Real("archive_epsilon", 1e-4, 0.1, log=True)
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
                archive_prune_policy_param,
                archive_epsilon_param,
            ],
        ),
    ]
    return AlgorithmConfigSpace("rvea", params, conditionals)


__all__ = ["build_rvea_config_space"]
