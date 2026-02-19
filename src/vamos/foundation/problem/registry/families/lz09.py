from __future__ import annotations

from ...lz import (
    LZ09F1Problem,
    LZ09F2Problem,
    LZ09F3Problem,
    LZ09F4Problem,
    LZ09F5Problem,
    LZ09F6Problem,
    LZ09F7Problem,
    LZ09F8Problem,
    LZ09F9Problem,
)
from ..common import ProblemSpec

SPECS: dict[str, ProblemSpec] = {}


def get_specs() -> dict[str, ProblemSpec]:
    if SPECS:
        return SPECS
    SPECS.update(
        {
            "lz09_f1": ProblemSpec(
                key="lz09_f1",
                label="LZ09 F1",
                default_n_var=10,
                default_n_obj=2,
                allow_n_obj_override=False,
                description="Li & Zhang LZ09 F1 benchmark.",
                factory=lambda n_var, _n_obj: LZ09F1Problem(n_var=n_var),
            ),
            "lz09_f2": ProblemSpec(
                key="lz09_f2",
                label="LZ09 F2",
                default_n_var=30,
                default_n_obj=2,
                allow_n_obj_override=False,
                description="Li & Zhang LZ09 F2 benchmark.",
                factory=lambda n_var, _n_obj: LZ09F2Problem(n_var=n_var),
            ),
            "lz09_f3": ProblemSpec(
                key="lz09_f3",
                label="LZ09 F3",
                default_n_var=30,
                default_n_obj=2,
                allow_n_obj_override=False,
                description="Li & Zhang LZ09 F3 benchmark.",
                factory=lambda n_var, _n_obj: LZ09F3Problem(n_var=n_var),
            ),
            "lz09_f4": ProblemSpec(
                key="lz09_f4",
                label="LZ09 F4",
                default_n_var=30,
                default_n_obj=2,
                allow_n_obj_override=False,
                description="Li & Zhang LZ09 F4 benchmark.",
                factory=lambda n_var, _n_obj: LZ09F4Problem(n_var=n_var),
            ),
            "lz09_f5": ProblemSpec(
                key="lz09_f5",
                label="LZ09 F5",
                default_n_var=30,
                default_n_obj=2,
                allow_n_obj_override=False,
                description="Li & Zhang LZ09 F5 benchmark.",
                factory=lambda n_var, _n_obj: LZ09F5Problem(n_var=n_var),
            ),
            "lz09_f6": ProblemSpec(
                key="lz09_f6",
                label="LZ09 F6",
                default_n_var=10,
                default_n_obj=3,
                allow_n_obj_override=False,
                description="Li & Zhang LZ09 F6 benchmark (tri-objective).",
                factory=lambda n_var, _n_obj: LZ09F6Problem(n_var=n_var),
            ),
            "lz09_f7": ProblemSpec(
                key="lz09_f7",
                label="LZ09 F7",
                default_n_var=10,
                default_n_obj=2,
                allow_n_obj_override=False,
                description="Li & Zhang LZ09 F7 benchmark.",
                factory=lambda n_var, _n_obj: LZ09F7Problem(n_var=n_var),
            ),
            "lz09_f8": ProblemSpec(
                key="lz09_f8",
                label="LZ09 F8",
                default_n_var=10,
                default_n_obj=2,
                allow_n_obj_override=False,
                description="Li & Zhang LZ09 F8 benchmark.",
                factory=lambda n_var, _n_obj: LZ09F8Problem(n_var=n_var),
            ),
            "lz09_f9": ProblemSpec(
                key="lz09_f9",
                label="LZ09 F9",
                default_n_var=30,
                default_n_obj=2,
                allow_n_obj_override=False,
                description="Li & Zhang LZ09 F9 benchmark.",
                factory=lambda n_var, _n_obj: LZ09F9Problem(n_var=n_var),
            ),
        }
    )
    return SPECS


__all__ = ["SPECS", "get_specs"]
