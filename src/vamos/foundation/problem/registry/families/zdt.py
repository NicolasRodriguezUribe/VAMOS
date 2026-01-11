from __future__ import annotations

from typing import Optional

from ...zdt1 import ZDT1Problem
from ...zdt2 import ZDT2Problem
from ...zdt3 import ZDT3Problem
from ...zdt4 import ZDT4Problem
from ...zdt6 import ZDT6Problem
from ..common import ProblemSpec


def _zdt1_factory(n_var: int, _ignored: Optional[int] = None) -> ZDT1Problem:
    return ZDT1Problem(n_var=n_var)


SPECS: dict[str, ProblemSpec] = {}


def get_specs() -> dict[str, ProblemSpec]:
    if SPECS:
        return SPECS
    SPECS.update(
        {
            "zdt1": ProblemSpec(
                key="zdt1",
                label="ZDT1",
                default_n_var=30,
                default_n_obj=2,
                allow_n_obj_override=False,
                description="Classic bi-objective benchmark with a convex Pareto front.",
                factory=_zdt1_factory,
            ),
            "zdt2": ProblemSpec(
                key="zdt2",
                label="ZDT2",
                default_n_var=30,
                default_n_obj=2,
                allow_n_obj_override=False,
                description="ZDT variant with a concave Pareto front.",
                factory=lambda n_var, _n_obj: ZDT2Problem(n_var=n_var),
            ),
            "zdt3": ProblemSpec(
                key="zdt3",
                label="ZDT3",
                default_n_var=30,
                default_n_obj=2,
                allow_n_obj_override=False,
                description="ZDT benchmark with a disconnected Pareto front.",
                factory=lambda n_var, _n_obj: ZDT3Problem(n_var=n_var),
            ),
            "zdt4": ProblemSpec(
                key="zdt4",
                label="ZDT4",
                default_n_var=10,
                default_n_obj=2,
                allow_n_obj_override=False,
                description="ZDT benchmark with multimodal landscape and mixed bounds.",
                factory=lambda n_var, _n_obj: ZDT4Problem(n_var=n_var),
            ),
            "zdt6": ProblemSpec(
                key="zdt6",
                label="ZDT6",
                default_n_var=10,
                default_n_obj=2,
                allow_n_obj_override=False,
                description="ZDT benchmark with non-uniform density over the Pareto set.",
                factory=lambda n_var, _n_obj: ZDT6Problem(n_var=n_var),
            ),
        }
    )
    return SPECS


__all__ = ["SPECS", "get_specs"]
