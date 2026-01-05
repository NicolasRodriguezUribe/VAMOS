"""Algorithm configuration module.

This package provides configuration dataclasses and fluent builders for all
multi-objective optimization algorithms.

Examples:
    from vamos.engine.algorithm.config import NSGAIIConfig, MOEADConfig

    # Fluent builder
    cfg = NSGAIIConfig().pop_size(100).crossover("sbx", prob=0.9).fixed()

    # Quick defaults
    cfg = NSGAIIConfig.default(pop_size=100, n_var=30)
"""

from .nsgaii import NSGAIIConfig, NSGAIIConfigData
from .moead import MOEADConfig, MOEADConfigData
from .spea2 import SPEA2Config, SPEA2ConfigData
from .ibea import IBEAConfig, IBEAConfigData
from .smsemoa import SMSEMOAConfig, SMSEMOAConfigData
from .smpso import SMPSOConfig, SMPSOConfigData
from .nsgaiii import NSGAIIIConfig, NSGAIIIConfigData
from .agemoea import AGEMOEAConfig, AGEMOEAConfigData
from .rvea import RVEAConfig, RVEAConfigData

__all__ = [
    # NSGA-II
    "NSGAIIConfig",
    "NSGAIIConfigData",
    # MOEA/D
    "MOEADConfig",
    "MOEADConfigData",
    # SPEA2
    "SPEA2Config",
    "SPEA2ConfigData",
    # IBEA
    "IBEAConfig",
    "IBEAConfigData",
    # SMS-EMOA
    "SMSEMOAConfig",
    "SMSEMOAConfigData",
    # SMPSO
    "SMPSOConfig",
    "SMPSOConfigData",
    # NSGA-III
    "NSGAIIIConfig",
    "NSGAIIIConfigData",
    # AGE-MOEA
    "AGEMOEAConfig",
    "AGEMOEAConfigData",
    # RVEA
    "RVEAConfig",
    "RVEAConfigData",
]
