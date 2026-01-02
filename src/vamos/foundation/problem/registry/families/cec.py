from __future__ import annotations

from ...cec import CEC2009CF1Problem, CEC2009UF1Problem, CEC2009UF2Problem, CEC2009UF3Problem
from ..common import ProblemSpec


SPECS: dict[str, ProblemSpec] = {}


def get_specs() -> dict[str, ProblemSpec]:
    if SPECS:
        return SPECS
    SPECS.update(
        {
            "cec2009_uf1": ProblemSpec(
                key="cec2009_uf1",
                label="CEC2009 UF1",
                default_n_var=30,
                default_n_obj=2,
                allow_n_obj_override=False,
                description="CEC2009 unconstrained function UF1.",
                factory=lambda n_var, _n_obj: CEC2009UF1Problem(n_var=n_var),
            ),
            "cec2009_uf2": ProblemSpec(
                key="cec2009_uf2",
                label="CEC2009 UF2",
                default_n_var=30,
                default_n_obj=2,
                allow_n_obj_override=False,
                description="CEC2009 unconstrained function UF2.",
                factory=lambda n_var, _n_obj: CEC2009UF2Problem(n_var=n_var),
            ),
            "cec2009_uf3": ProblemSpec(
                key="cec2009_uf3",
                label="CEC2009 UF3",
                default_n_var=30,
                default_n_obj=2,
                allow_n_obj_override=False,
                description="CEC2009 unconstrained function UF3.",
                factory=lambda n_var, _n_obj: CEC2009UF3Problem(n_var=n_var),
            ),
            "cec2009_cf1": ProblemSpec(
                key="cec2009_cf1",
                label="CEC2009 CF1",
                default_n_var=30,
                default_n_obj=2,
                allow_n_obj_override=False,
                description="CEC2009 constrained function CF1.",
                factory=lambda n_var, _n_obj: CEC2009CF1Problem(n_var=n_var),
            ),
        }
    )
    return SPECS


__all__ = ["SPECS", "get_specs"]
