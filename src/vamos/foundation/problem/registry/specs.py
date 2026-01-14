from __future__ import annotations

from typing import TYPE_CHECKING

from .common import ProblemFactory, ProblemSpec
from .families import cec, dtlz, lz09, misc, real_world, wfg, zdt

_PROBLEM_SPECS: dict[str, ProblemSpec] | None = None

if TYPE_CHECKING:
    PROBLEM_SPECS: dict[str, ProblemSpec]


def _build_problem_specs() -> dict[str, ProblemSpec]:
    specs: dict[str, ProblemSpec] = {}
    for family in (
        zdt.get_specs(),
        dtlz.get_specs(),
        lz09.get_specs(),
        cec.get_specs(),
        wfg.get_specs(),
        misc.get_specs(),
        real_world.get_specs(),
    ):
        specs.update(family)
    return specs


def get_problem_specs() -> dict[str, ProblemSpec]:
    global _PROBLEM_SPECS
    if _PROBLEM_SPECS is None:
        _PROBLEM_SPECS = _build_problem_specs()
    return _PROBLEM_SPECS


def available_problem_names() -> tuple[str, ...]:
    return tuple(get_problem_specs().keys())


def __getattr__(name: str) -> object:
    if name == "PROBLEM_SPECS":
        return get_problem_specs()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({"PROBLEM_SPECS", "get_problem_specs", "available_problem_names"} | set(globals()))


__all__ = ["ProblemSpec", "ProblemFactory", "PROBLEM_SPECS", "available_problem_names", "get_problem_specs"]
