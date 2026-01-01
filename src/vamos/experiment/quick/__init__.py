"""Quick-start API for one-liner experiments."""

from .api import run_moead, run_nsgaii, run_nsgaiii, run_smsemoa, run_spea2
from .result import QuickResult

__all__ = [
    "QuickResult",
    "run_nsgaii",
    "run_moead",
    "run_spea2",
    "run_smsemoa",
    "run_nsgaiii",
]
