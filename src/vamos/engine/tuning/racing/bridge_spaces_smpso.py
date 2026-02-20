"""
SMPSO configuration space builders for tuning.

SMPSO uses particle swarm (velocity-based) movement rather than traditional
crossover/mutation. The mixed variant exposes mutation-only tuning.
"""

from __future__ import annotations

from .config_space import AlgorithmConfigSpace, SpacePart, compose_config_space
from .param_space import Categorical, Int, ParamType, Real

# ---------------------------------------------------------------------------
# SMPSO has a single encoding; no core/operator split needed.
# ---------------------------------------------------------------------------


def _smpso_part() -> SpacePart:
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
    return params, [], []


def _smpso_mixed_part() -> SpacePart:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True),
        Int("archive_size", 20, 200, log=True),
        Real("inertia", 0.1, 0.9),
        Real("c1", 0.5, 2.5),
        Real("c2", 0.5, 2.5),
        Real("vmax_fraction", 0.1, 1.0),
        Categorical("mutation", ["mixed", "gaussian"]),
        Real("mutation_prob", 0.01, 0.5),
    ]
    return params, [], []


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def build_smpso_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("smpso", _smpso_part())


def build_smpso_mixed_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("smpso_mixed", _smpso_mixed_part())


__all__ = ["build_smpso_config_space", "build_smpso_mixed_config_space"]
