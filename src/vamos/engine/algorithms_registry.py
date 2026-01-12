"""
Convenience access to built-in multi-objective algorithms and their configs.

For advanced customization, drop down to `vamos.engine.algorithm.*`.
"""

from __future__ import annotations

from typing import Tuple

from vamos.foundation.encoding import normalize_encoding
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
from vamos.engine.algorithm.nsgaiii import NSGAIII
from vamos.engine.algorithm.nsgaii import NSGAII
from vamos.engine.algorithm.registry import get_algorithms_registry, resolve_algorithm
from vamos.engine.algorithm.smpso import SMPSO
from vamos.engine.algorithm.smsemoa import SMSEMOA
from vamos.engine.algorithm.spea2 import SPEA2
from vamos.engine.algorithm.ibea import IBEA


from vamos.engine.algorithm.components.variation.helpers import (
    PERM_CROSSOVER,
    PERM_MUTATION,
    BINARY_CROSSOVER,
    BINARY_MUTATION,
    INT_CROSSOVER,
    INT_MUTATION,
    MIXED_CROSSOVER,
    MIXED_MUTATION,
    REAL_CROSSOVER,
    REAL_MUTATION,
)


def available_algorithms() -> Tuple[str, ...]:
    """Return the canonical algorithm identifiers supported by the engine."""
    return tuple(sorted(get_algorithms_registry().keys()))


def available_crossover_methods(encoding: str = "real") -> Tuple[str, ...]:
    """
    Return the available crossover method identifiers for a given encoding.

    Args:
        encoding: Variable encoding type ("real", "binary", "permutation", "integer", "mixed").

    Returns:
        Tuple of supported crossover method strings.
    """
    normalized = normalize_encoding(encoding)
    if normalized == "real":
        return tuple(sorted(REAL_CROSSOVER.keys()))
    if normalized == "binary":
        return tuple(sorted(BINARY_CROSSOVER.keys()))
    if normalized == "permutation":
        return tuple(sorted(PERM_CROSSOVER.keys()))
    if normalized == "integer":
        return tuple(sorted(INT_CROSSOVER.keys()))
    if normalized == "mixed":
        return tuple(sorted(MIXED_CROSSOVER.keys()))
    return ()


def available_mutation_methods(encoding: str = "real") -> Tuple[str, ...]:
    """
    Return the available mutation method identifiers for a given encoding.

    Args:
        encoding: Variable encoding type ("real", "binary", "permutation", "integer", "mixed").

    Returns:
        Tuple of supported mutation method strings.
    """
    normalized = normalize_encoding(encoding)
    if normalized == "real":
        return tuple(sorted(REAL_MUTATION.keys()))
    if normalized == "binary":
        return tuple(sorted(BINARY_MUTATION.keys()))
    if normalized == "permutation":
        return tuple(sorted(PERM_MUTATION.keys()))
    if normalized == "integer":
        return tuple(sorted(INT_MUTATION.keys()))
    if normalized == "mixed":
        return tuple(sorted(MIXED_MUTATION.keys()))
    return ()


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
    "available_crossover_methods",
    "available_mutation_methods",
    "resolve_algorithm",
]
