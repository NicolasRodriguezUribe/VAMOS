from __future__ import annotations

from .config_space import SpacePart
from .param_space import Categorical, ConditionalBlock, Int, ParamType, Real


def real_operator_part_medium(
    *,
    mutation_prob_param: str = "mutation_prob",
    mutation_prob_bounds: tuple[float, float] = (0.01, 0.5),
    crossover_prob_bounds: tuple[float, float] = (0.6, 1.0),
    include_initializer: bool = True,
    include_repair: bool = True,
) -> SpacePart:
    params: list[ParamType] = []
    if include_initializer:
        params.append(Categorical("initializer", ["random", "lhs", "scatter"]))
    params.extend(
        [
            Categorical("crossover", ["sbx", "blx_alpha", "arithmetic", "pcx", "undx", "simplex"]),
            Real("crossover_prob", crossover_prob_bounds[0], crossover_prob_bounds[1]),
            Categorical(
                "mutation",
                ["pm", "linked_polynomial", "non_uniform", "gaussian", "uniform_reset", "cauchy", "uniform"],
            ),
            Real(mutation_prob_param, mutation_prob_bounds[0], mutation_prob_bounds[1]),
            Real("mutation_eta", 5.0, 40.0),
        ]
    )
    if include_repair:
        params.append(Categorical("repair", ["none", "clip", "reflect", "random", "round"]))

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
    ]
    if include_initializer:
        conditionals.append(
            ConditionalBlock(
                "initializer",
                "scatter",
                [Categorical("scatter_base_size_factor", [0.1, 0.2, 0.3, 0.5, 0.75, 1.0])],
            ),
        )
    return params, conditionals, []


def permutation_operator_part_full(
    *,
    mutation_prob_param: str = "mutation_prob",
    mutation_prob_bounds: tuple[float, float] = (0.01, 0.5),
    crossover_prob_bounds: tuple[float, float] = (0.6, 1.0),
) -> SpacePart:
    params: list[ParamType] = [
        Categorical("crossover", ["ox", "pmx", "edge", "cycle", "position", "aex"]),
        Real("crossover_prob", crossover_prob_bounds[0], crossover_prob_bounds[1]),
        Categorical("mutation", ["swap", "insert", "scramble", "inversion", "displacement", "two_opt"]),
        Real(mutation_prob_param, mutation_prob_bounds[0], mutation_prob_bounds[1]),
    ]
    return params, [], []


def binary_operator_part_full(
    *,
    mutation_prob_param: str = "mutation_prob",
    mutation_prob_bounds: tuple[float, float] = (0.01, 0.5),
    crossover_prob_bounds: tuple[float, float] = (0.6, 1.0),
) -> SpacePart:
    params: list[ParamType] = [
        Categorical("crossover", ["hux", "uniform", "one_point", "two_point"]),
        Real("crossover_prob", crossover_prob_bounds[0], crossover_prob_bounds[1]),
        Categorical("mutation", ["bitflip", "segment_inversion"]),
        Real(mutation_prob_param, mutation_prob_bounds[0], mutation_prob_bounds[1]),
    ]
    return params, [], []


def integer_operator_part_full(
    *,
    mutation_prob_param: str = "mutation_prob",
    mutation_prob_bounds: tuple[float, float] = (0.01, 0.5),
    crossover_prob_bounds: tuple[float, float] = (0.6, 1.0),
) -> SpacePart:
    params: list[ParamType] = [
        Categorical("crossover", ["uniform", "arithmetic", "sbx"]),
        Real("crossover_prob", crossover_prob_bounds[0], crossover_prob_bounds[1]),
        Categorical("mutation", ["reset", "creep", "pm", "gaussian", "boundary"]),
        Real(mutation_prob_param, mutation_prob_bounds[0], mutation_prob_bounds[1]),
    ]
    conditionals = [
        ConditionalBlock("crossover", "sbx", [Real("crossover_eta", 5.0, 40.0)]),
        ConditionalBlock("mutation", "pm", [Real("mutation_eta", 5.0, 40.0)]),
        ConditionalBlock("mutation", "creep", [Int("creep_step", 1, 5)]),
        ConditionalBlock("mutation", "gaussian", [Real("gaussian_sigma", 0.1, 5.0)]),
    ]
    return params, conditionals, []


def mixed_operator_part(
    *,
    crossover_choices: tuple[str, ...] = ("mixed",),
    mutation_choices: tuple[str, ...] = ("mixed",),
    mutation_prob_param: str = "mutation_prob",
    mutation_prob_bounds: tuple[float, float] = (0.01, 0.5),
    crossover_prob_bounds: tuple[float, float] = (0.6, 1.0),
) -> SpacePart:
    params: list[ParamType] = [
        Categorical("crossover", list(crossover_choices)),
        Real("crossover_prob", crossover_prob_bounds[0], crossover_prob_bounds[1]),
        Categorical("mutation", list(mutation_choices)),
        Real(mutation_prob_param, mutation_prob_bounds[0], mutation_prob_bounds[1]),
    ]
    return params, [], []


__all__ = [
    "binary_operator_part_full",
    "integer_operator_part_full",
    "mixed_operator_part",
    "permutation_operator_part_full",
    "real_operator_part_medium",
]
