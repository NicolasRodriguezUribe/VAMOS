"""Problem definitions for multi-objective optimization.

This module contains various test problems for multi-objective optimization,
including ZDT, DTLZ, and WFG problems.
"""

from .zdt1 import ZDT1Problem
from .dtlz import DTLZ1Problem, DTLZ2Problem, DTLZ3Problem, DTLZ4Problem
from .wfg import (
    WFG1Problem,
    WFG2Problem,
    WFG3Problem,
    WFG4Problem,
    WFG5Problem,
    WFG6Problem,
    WFG7Problem,
    WFG8Problem,
    WFG9Problem,
)

__all__ = [
    "ZDT1Problem",
    "DTLZ1Problem",
    "DTLZ2Problem",
    "DTLZ3Problem",
    "DTLZ4Problem",
    "WFG1Problem",
    "WFG2Problem",
    "WFG3Problem",
    "WFG4Problem",
    "WFG5Problem",
    "WFG6Problem",
    "WFG7Problem",
    "WFG8Problem",
    "WFG9Problem",
]
