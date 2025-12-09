"""
Problem registry: specs, selection, and factories.
"""

from .specs import ProblemSpec, PROBLEM_SPECS, available_problem_names  # noqa: F401
from .selection import ProblemSelection, make_problem_selection  # noqa: F401

__all__ = ["ProblemSpec", "ProblemSelection", "PROBLEM_SPECS", "available_problem_names", "make_problem_selection"]
