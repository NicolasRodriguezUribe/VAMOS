"""
Convenient access to benchmark problems and registry helpers.

For the full catalog or custom problems, import from `vamos.foundation.problem`.
"""
from __future__ import annotations

from vamos.foundation.problem.dtlz import DTLZ1Problem as DTLZ1
from vamos.foundation.problem.dtlz import DTLZ2Problem as DTLZ2
from vamos.foundation.problem.dtlz import DTLZ3Problem as DTLZ3
from vamos.foundation.problem.dtlz import DTLZ4Problem as DTLZ4
from vamos.foundation.problem.registry import (
    ProblemSelection,
    ProblemSpec,
    available_problem_names,
    make_problem_selection,
)
from vamos.foundation.problem.wfg import (
    WFG1Problem as WFG1,
    WFG2Problem as WFG2,
    WFG3Problem as WFG3,
    WFG4Problem as WFG4,
    WFG5Problem as WFG5,
    WFG6Problem as WFG6,
    WFG7Problem as WFG7,
    WFG8Problem as WFG8,
    WFG9Problem as WFG9,
)
from vamos.foundation.problem.zdt1 import ZDT1Problem as ZDT1
from vamos.foundation.problem.zdt2 import ZDT2Problem as ZDT2
from vamos.foundation.problem.zdt3 import ZDT3Problem as ZDT3
from vamos.foundation.problem.zdt4 import ZDT4Problem as ZDT4
from vamos.foundation.problem.zdt6 import ZDT6Problem as ZDT6

# Real-world / application problems
from vamos.foundation.problem.real_world.feature_selection import (
    FeatureSelectionProblem,
)
from vamos.foundation.problem.real_world.hyperparam import (
    HyperparameterTuningProblem,
)
from vamos.foundation.problem.real_world.engineering import (
    WeldedBeamDesignProblem,
)

# Convenience alias for users expecting a simple accessor name.
get_problem_names = available_problem_names

__all__ = [
    # ZDT family
    "ZDT1",
    "ZDT2",
    "ZDT3",
    "ZDT4",
    "ZDT6",
    # DTLZ family
    "DTLZ1",
    "DTLZ2",
    "DTLZ3",
    "DTLZ4",
    # WFG family
    "WFG1",
    "WFG2",
    "WFG3",
    "WFG4",
    "WFG5",
    "WFG6",
    "WFG7",
    "WFG8",
    "WFG9",
    # Real-world problems
    "FeatureSelectionProblem",
    "HyperparameterTuningProblem",
    "WeldedBeamDesignProblem",
    # Registry helpers
    "ProblemSpec",
    "ProblemSelection",
    "available_problem_names",
    "get_problem_names",
    "make_problem_selection",
]
