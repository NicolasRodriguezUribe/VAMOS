"""Algorithm configuration module.

This package provides configuration dataclasses and fluent builders for all
multi-objective optimization algorithms.

Examples:
    from vamos.engine.algorithm.config import NSGAIIConfig, MOEADConfig

    # Fluent builder
    cfg = NSGAIIConfig().pop_size(100).crossover("sbx", prob=1.0).fixed()

    # Quick defaults
    cfg = NSGAIIConfig.default(pop_size=100, n_var=30)
"""

from .types import AlgorithmConfigDict, AlgorithmConfigMapping, AlgorithmConfigProtocol
from .generic import GenericAlgorithmConfig
from .nsgaii import NSGAIIConfig, NSGAIIConfigData, NSGAIIConfigDict
from .moead import MOEADConfig, MOEADConfigData, MOEADConfigDict
from .spea2 import SPEA2Config, SPEA2ConfigData, SPEA2ConfigDict
from .ibea import IBEAConfig, IBEAConfigData, IBEAConfigDict
from .smsemoa import SMSEMOAConfig, SMSEMOAConfigData, SMSEMOAConfigDict
from .smpso import SMPSOConfig, SMPSOConfigData, SMPSOConfigDict
from .nsgaiii import NSGAIIIConfig, NSGAIIIConfigData, NSGAIIIConfigDict
from .agemoea import AGEMOEAConfig, AGEMOEAConfigData, AGEMOEAConfigDict
from .rvea import RVEAConfig, RVEAConfigData, RVEAConfigDict

__all__ = [
    # NSGA-II
    "NSGAIIConfig",
    "NSGAIIConfigData",
    "NSGAIIConfigDict",
    # MOEA/D
    "MOEADConfig",
    "MOEADConfigData",
    "MOEADConfigDict",
    # SPEA2
    "SPEA2Config",
    "SPEA2ConfigData",
    "SPEA2ConfigDict",
    # IBEA
    "IBEAConfig",
    "IBEAConfigData",
    "IBEAConfigDict",
    # SMS-EMOA
    "SMSEMOAConfig",
    "SMSEMOAConfigData",
    "SMSEMOAConfigDict",
    # SMPSO
    "SMPSOConfig",
    "SMPSOConfigData",
    "SMPSOConfigDict",
    # NSGA-III
    "NSGAIIIConfig",
    "NSGAIIIConfigData",
    "NSGAIIIConfigDict",
    # AGE-MOEA
    "AGEMOEAConfig",
    "AGEMOEAConfigData",
    "AGEMOEAConfigDict",
    # RVEA
    "RVEAConfig",
    "RVEAConfigData",
    "RVEAConfigDict",
    # Plugin/custom
    "GenericAlgorithmConfig",
    # Typing helpers
    "AlgorithmConfigDict",
    "AlgorithmConfigMapping",
    "AlgorithmConfigProtocol",
]
