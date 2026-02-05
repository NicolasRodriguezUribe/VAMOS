"""
NSGA-III configuration space builders for tuning.
"""

from __future__ import annotations

from .config_space import AlgorithmConfigSpace
from .param_space import Categorical, ConditionalBlock, Int, ParamType, Real


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


def build_nsgaiii_binary_config_space() -> AlgorithmConfigSpace:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True),
        Categorical("crossover", ["hux", "uniform", "one_point", "two_point"]),
        Real("crossover_prob", 0.6, 1.0),
        Categorical("mutation", ["bitflip"]),
        Real("mutation_prob", 0.01, 0.5),
        Int("selection_pressure", 2, 10),
    ]
    return AlgorithmConfigSpace("nsgaiii_binary", params, [])


def build_nsgaiii_integer_config_space() -> AlgorithmConfigSpace:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True),
        Categorical("crossover", ["uniform", "arithmetic", "sbx"]),
        Real("crossover_prob", 0.6, 1.0),
        Categorical("mutation", ["reset", "creep", "pm"]),
        Real("mutation_prob", 0.01, 0.5),
        Int("selection_pressure", 2, 10),
    ]
    conditionals = [
        ConditionalBlock("crossover", "sbx", [Real("crossover_eta", 5.0, 40.0)]),
        ConditionalBlock("mutation", "pm", [Real("mutation_eta", 5.0, 40.0)]),
        ConditionalBlock("mutation", "creep", [Int("creep_step", 1, 5)]),
    ]
    return AlgorithmConfigSpace("nsgaiii_integer", params, conditionals)


__all__ = [
    "build_nsgaiii_config_space",
    "build_nsgaiii_binary_config_space",
    "build_nsgaiii_integer_config_space",
]
