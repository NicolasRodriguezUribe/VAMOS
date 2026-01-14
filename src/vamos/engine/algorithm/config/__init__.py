"""Algorithm configuration module.

This package provides configuration dataclasses and fluent builders for all
multi-objective optimization algorithms.

Examples:
    from vamos.engine.algorithm.config import NSGAIIConfig, MOEADConfig

    # Fluent builder
    cfg = NSGAIIConfig.builder().pop_size(100).crossover("sbx", prob=1.0).build()

    # Quick defaults
    cfg = NSGAIIConfig.default(pop_size=100, n_var=30)
"""

from .types import AlgorithmConfigMapping, AlgorithmConfigProtocol
from .generic import GenericAlgorithmConfig
from .nsgaii import NSGAIIConfig
from .moead import MOEADConfig
from .spea2 import SPEA2Config
from .ibea import IBEAConfig
from .smsemoa import SMSEMOAConfig
from .smpso import SMPSOConfig
from .nsgaiii import NSGAIIIConfig
from .agemoea import AGEMOEAConfig
from .rvea import RVEAConfig

__all__ = [
    # NSGA-II
    "NSGAIIConfig",
    # MOEA/D
    "MOEADConfig",
    # SPEA2
    "SPEA2Config",
    # IBEA
    "IBEAConfig",
    # SMS-EMOA
    "SMSEMOAConfig",
    # SMPSO
    "SMPSOConfig",
    # NSGA-III
    "NSGAIIIConfig",
    # AGE-MOEA
    "AGEMOEAConfig",
    # RVEA
    "RVEAConfig",
    # Plugin/custom
    "GenericAlgorithmConfig",
    # Typing helpers
    "AlgorithmConfigMapping",
    "AlgorithmConfigProtocol",
]
