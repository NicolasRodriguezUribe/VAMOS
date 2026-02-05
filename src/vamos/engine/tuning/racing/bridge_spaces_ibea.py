"""
IBEA configuration space builders for tuning.
"""

from __future__ import annotations

from .config_space import AlgorithmConfigSpace
from .param_space import Categorical, ConditionalBlock, Int, ParamType, Real


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


def build_ibea_binary_config_space() -> AlgorithmConfigSpace:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True),
        Categorical("crossover", ["hux", "uniform", "one_point", "two_point"]),
        Real("crossover_prob", 0.6, 1.0),
        Categorical("mutation", ["bitflip"]),
        Real("mutation_prob", 0.01, 0.5),
        Int("selection_pressure", 2, 10),
        Categorical("indicator", ["eps", "hypervolume"]),
        Real("kappa", 0.01, 0.2),
    ]
    return AlgorithmConfigSpace("ibea_binary", params, [])


def build_ibea_integer_config_space() -> AlgorithmConfigSpace:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True),
        Categorical("crossover", ["uniform", "arithmetic", "sbx"]),
        Real("crossover_prob", 0.6, 1.0),
        Categorical("mutation", ["reset", "creep", "pm"]),
        Real("mutation_prob", 0.01, 0.5),
        Int("selection_pressure", 2, 10),
        Categorical("indicator", ["eps", "hypervolume"]),
        Real("kappa", 0.01, 0.2),
    ]
    conditionals = [
        ConditionalBlock("crossover", "sbx", [Real("crossover_eta", 5.0, 40.0)]),
        ConditionalBlock("mutation", "pm", [Real("mutation_eta", 5.0, 40.0)]),
        ConditionalBlock("mutation", "creep", [Int("creep_step", 1, 5)]),
    ]
    return AlgorithmConfigSpace("ibea_integer", params, conditionals)


__all__ = [
    "build_ibea_config_space",
    "build_ibea_binary_config_space",
    "build_ibea_integer_config_space",
]
