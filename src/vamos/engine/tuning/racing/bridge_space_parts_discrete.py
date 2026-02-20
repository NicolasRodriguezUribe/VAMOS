from __future__ import annotations

from .config_space import SpacePart
from .param_space import Categorical, ConditionalBlock, Int, ParamType, Real


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
]
