"""
SPEA2 configuration space builders for tuning.
"""

from __future__ import annotations

from .config_space import AlgorithmConfigSpace
from .param_space import Categorical, Int, ParamType, Real


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


__all__ = ["build_spea2_config_space"]
