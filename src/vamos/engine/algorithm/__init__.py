"""
Engine layer: multi-objective algorithm implementations.

This package contains concrete algorithms (e.g. NSGA-II/III, MOEA/D, SMPSO, etc.)
and shared building blocks under `vamos.engine.algorithm.components`.

Algorithms
----------
- NSGAII: Non-dominated Sorting Genetic Algorithm II
- NSGAIII: Reference-point based NSGA-III
- MOEAD: Multi-Objective Evolutionary Algorithm based on Decomposition
- SPEA2: Strength Pareto Evolutionary Algorithm 2
- SMSEMOA: S-Metric Selection EMOA (hypervolume-based)
- IBEA: Indicator-Based Evolutionary Algorithm
- SMPSO: Speed-constrained Multi-objective PSO
- AGEMOEA: Adaptive Geometry Estimation MOEA
- RVEA: Reference Vector Guided Evolutionary Algorithm

Usage
-----
Algorithms are typically accessed through the registry or factory:

    from vamos.engine.algorithm import NSGAII, MOEAD, SPEA2, AGEMOEA, RVEA
    from vamos.engine.algorithm.registry import get_algorithm
    from vamos.engine.algorithm.factory import create_algorithm

For configuration builders:

    from vamos.engine.algorithm.config import NSGAIIConfig, MOEADConfig
"""

from vamos.engine.algorithm.nsgaii import NSGAII
from vamos.engine.algorithm.moead import MOEAD
from vamos.engine.algorithm.spea2 import SPEA2
from vamos.engine.algorithm.smsemoa import SMSEMOA
from vamos.engine.algorithm.nsgaiii import NSGAIII
from vamos.engine.algorithm.ibea import IBEA
from vamos.engine.algorithm.smpso import SMPSO
from vamos.engine.algorithm.agemoea import AGEMOEA
from vamos.engine.algorithm.rvea import RVEA

from vamos.engine.algorithm.components.protocol import (
    AlgorithmProtocol,
    InteractiveAlgorithmProtocol,
    SelectionMethod,
    SurvivalMethod,
    ConstraintMode,
)

__all__ = [
    # Algorithms
    "NSGAII",
    "NSGAIII",
    "MOEAD",
    "SPEA2",
    "SMSEMOA",
    "IBEA",
    "SMPSO",
    "AGEMOEA",
    "RVEA",
    # Protocols and enums
    "AlgorithmProtocol",
    "InteractiveAlgorithmProtocol",
    "SelectionMethod",
    "SurvivalMethod",
    "ConstraintMode",
]
