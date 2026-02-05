"""
SMPSO configuration space builders for tuning.
"""

from __future__ import annotations

from .config_space import AlgorithmConfigSpace
from .param_space import Int, ParamType, Real


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


__all__ = ["build_smpso_config_space"]
