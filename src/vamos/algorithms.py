"""
Convenience access to built-in multi-objective algorithms and their configs.

For advanced customization, drop down to `vamos.engine.algorithm.*`.
"""
from __future__ import annotations

from typing import Tuple

from vamos.engine.algorithm.config import (
    IBEAConfig,
    IBEAConfigData,
    MOEADConfig,
    MOEADConfigData,
    NSGAIIConfig,
    NSGAIIConfigData,
    NSGAIIIConfig,
    NSGAIIIConfigData,
    SMPSOConfig,
    SMPSOConfigData,
    SMSEMOAConfig,
    SMSEMOAConfigData,
    SPEA2Config,
    SPEA2ConfigData,
)
from vamos.engine.algorithm.moead import MOEAD
from vamos.engine.algorithm.nsga3 import NSGAIII
from vamos.engine.algorithm.nsgaii import NSGAII
from vamos.engine.algorithm.registry import ALGORITHMS, resolve_algorithm
from vamos.engine.algorithm.smpso import SMPSO
from vamos.engine.algorithm.smsemoa import SMSEMOA
from vamos.engine.algorithm.spea2 import SPEA2
from vamos.engine.algorithm.ibea import IBEA


def available_algorithms() -> Tuple[str, ...]:
    """Return the canonical algorithm identifiers supported by the engine."""
    return tuple(sorted(ALGORITHMS))


__all__ = [
    "NSGAII",
    "NSGAIII",
    "MOEAD",
    "SMSEMOA",
    "SPEA2",
    "IBEA",
    "SMPSO",
    "NSGAIIConfig",
    "NSGAIIConfigData",
    "MOEADConfig",
    "MOEADConfigData",
    "SMSEMOAConfig",
    "SMSEMOAConfigData",
    "NSGAIIIConfig",
    "NSGAIIIConfigData",
    "SPEA2Config",
    "SPEA2ConfigData",
    "IBEAConfig",
    "IBEAConfigData",
    "SMPSOConfig",
    "SMPSOConfigData",
    "available_algorithms",
    "resolve_algorithm",
]
