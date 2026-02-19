"""
Problem registry: specs, selection, and factories.
"""

from .selection import ProblemSelection, make_problem_selection  # noqa: F401
from .specs import ProblemSpec, available_problem_names, get_problem_specs  # noqa: F401


def __getattr__(name: str) -> object:
    if name == "PROBLEM_SPECS":
        return get_problem_specs()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({"PROBLEM_SPECS"} | set(globals()))


__all__ = [
    "ProblemSpec",
    "ProblemSelection",
    "PROBLEM_SPECS",
    "available_problem_names",
    "make_problem_selection",
    "get_problem_specs",
]
