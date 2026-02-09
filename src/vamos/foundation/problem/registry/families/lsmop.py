from __future__ import annotations

from collections.abc import Callable

from ...lsmop import LSMOP1, LSMOP2, LSMOP3, LSMOP4, LSMOP5, LSMOP6, LSMOP7, LSMOP8, LSMOP9
from ..common import ProblemSpec


def _lsmop_factory(cls: Callable[..., object], n_var: int, n_obj: int | None) -> object:
    return cls(n_var=n_var, n_obj=n_obj if n_obj is not None else 3)


SPECS: dict[str, ProblemSpec] = {}


def get_specs() -> dict[str, ProblemSpec]:
    if SPECS:
        return SPECS

    SPECS.update(
        {
            "lsmop1": ProblemSpec(
                key="lsmop1",
                label="LSMOP1",
                default_n_var=300,
                default_n_obj=3,
                allow_n_obj_override=True,
                description="Large-scale many-objective problem LSMOP1.",
                factory=lambda n_var, n_obj: _lsmop_factory(LSMOP1, n_var, n_obj),
            ),
            "lsmop2": ProblemSpec(
                key="lsmop2",
                label="LSMOP2",
                default_n_var=300,
                default_n_obj=3,
                allow_n_obj_override=True,
                description="Large-scale many-objective problem LSMOP2.",
                factory=lambda n_var, n_obj: _lsmop_factory(LSMOP2, n_var, n_obj),
            ),
            "lsmop3": ProblemSpec(
                key="lsmop3",
                label="LSMOP3",
                default_n_var=300,
                default_n_obj=3,
                allow_n_obj_override=True,
                description="Large-scale many-objective problem LSMOP3.",
                factory=lambda n_var, n_obj: _lsmop_factory(LSMOP3, n_var, n_obj),
            ),
            "lsmop4": ProblemSpec(
                key="lsmop4",
                label="LSMOP4",
                default_n_var=300,
                default_n_obj=3,
                allow_n_obj_override=True,
                description="Large-scale many-objective problem LSMOP4.",
                factory=lambda n_var, n_obj: _lsmop_factory(LSMOP4, n_var, n_obj),
            ),
            "lsmop5": ProblemSpec(
                key="lsmop5",
                label="LSMOP5",
                default_n_var=300,
                default_n_obj=3,
                allow_n_obj_override=True,
                description="Large-scale many-objective problem LSMOP5.",
                factory=lambda n_var, n_obj: _lsmop_factory(LSMOP5, n_var, n_obj),
            ),
            "lsmop6": ProblemSpec(
                key="lsmop6",
                label="LSMOP6",
                default_n_var=300,
                default_n_obj=3,
                allow_n_obj_override=True,
                description="Large-scale many-objective problem LSMOP6.",
                factory=lambda n_var, n_obj: _lsmop_factory(LSMOP6, n_var, n_obj),
            ),
            "lsmop7": ProblemSpec(
                key="lsmop7",
                label="LSMOP7",
                default_n_var=300,
                default_n_obj=3,
                allow_n_obj_override=True,
                description="Large-scale many-objective problem LSMOP7.",
                factory=lambda n_var, n_obj: _lsmop_factory(LSMOP7, n_var, n_obj),
            ),
            "lsmop8": ProblemSpec(
                key="lsmop8",
                label="LSMOP8",
                default_n_var=300,
                default_n_obj=3,
                allow_n_obj_override=True,
                description="Large-scale many-objective problem LSMOP8.",
                factory=lambda n_var, n_obj: _lsmop_factory(LSMOP8, n_var, n_obj),
            ),
            "lsmop9": ProblemSpec(
                key="lsmop9",
                label="LSMOP9",
                default_n_var=300,
                default_n_obj=3,
                allow_n_obj_override=True,
                description="Large-scale many-objective problem LSMOP9.",
                factory=lambda n_var, n_obj: _lsmop_factory(LSMOP9, n_var, n_obj),
            ),
        }
    )
    return SPECS


__all__ = ["SPECS", "get_specs"]
