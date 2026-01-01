from __future__ import annotations

from typing import Dict, Tuple

from .common import ProblemFactory, ProblemSpec
from .families import cec, dtlz, lz09, misc, real_world, wfg, zdt

PROBLEM_SPECS: Dict[str, ProblemSpec] = {}
for _family in (
    zdt.SPECS,
    dtlz.SPECS,
    lz09.SPECS,
    cec.SPECS,
    wfg.SPECS,
    misc.SPECS,
    real_world.SPECS,
):
    PROBLEM_SPECS.update(_family)


def available_problem_names() -> Tuple[str, ...]:
    return tuple(PROBLEM_SPECS.keys())


__all__ = ["ProblemSpec", "ProblemFactory", "PROBLEM_SPECS", "available_problem_names"]
