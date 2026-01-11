from __future__ import annotations

from typing import Callable, Optional

from ...dtlz import DTLZ1Problem, DTLZ2Problem, DTLZ3Problem, DTLZ4Problem, DTLZ7Problem
from ..common import ProblemSpec


def _dtlz_factory(cls: Callable[..., object], n_var: int, n_obj: Optional[int]) -> object:
    return cls(n_var=n_var, n_obj=n_obj if n_obj is not None else 3)


def _dtlz_k5_n_var(n_obj: int) -> int:
    return n_obj + 4


def _dtlz_k10_n_var(n_obj: int) -> int:
    return n_obj + 9


def _dtlz_k20_n_var(n_obj: int) -> int:
    return n_obj + 19


SPECS: dict[str, ProblemSpec] = {}


def get_specs() -> dict[str, ProblemSpec]:
    if SPECS:
        return SPECS
    SPECS.update(
        {
            "dtlz1": ProblemSpec(
                key="dtlz1",
                label="DTLZ1",
                default_n_var=7,
                default_n_obj=3,
                allow_n_obj_override=True,
                default_n_var_fn=_dtlz_k5_n_var,
                description="DTLZ1 with configurable objectives (defaults to 3).",
                factory=lambda n_var, n_obj: _dtlz_factory(DTLZ1Problem, n_var, n_obj),
            ),
            "dtlz2": ProblemSpec(
                key="dtlz2",
                label="DTLZ2",
                default_n_var=12,
                default_n_obj=3,
                allow_n_obj_override=True,
                default_n_var_fn=_dtlz_k10_n_var,
                description="DTLZ2 with configurable objectives (defaults to 3).",
                factory=lambda n_var, n_obj: _dtlz_factory(DTLZ2Problem, n_var, n_obj),
            ),
            "dtlz3": ProblemSpec(
                key="dtlz3",
                label="DTLZ3",
                default_n_var=12,
                default_n_obj=3,
                allow_n_obj_override=True,
                default_n_var_fn=_dtlz_k10_n_var,
                description="DTLZ3 with configurable objectives (defaults to 3).",
                factory=lambda n_var, n_obj: _dtlz_factory(DTLZ3Problem, n_var, n_obj),
            ),
            "dtlz4": ProblemSpec(
                key="dtlz4",
                label="DTLZ4",
                default_n_var=12,
                default_n_obj=3,
                allow_n_obj_override=True,
                default_n_var_fn=_dtlz_k10_n_var,
                description="DTLZ4 with configurable objectives (defaults to 3).",
                factory=lambda n_var, n_obj: _dtlz_factory(DTLZ4Problem, n_var, n_obj),
            ),
            "dtlz7": ProblemSpec(
                key="dtlz7",
                label="DTLZ7",
                default_n_var=22,
                default_n_obj=3,
                allow_n_obj_override=True,
                default_n_var_fn=_dtlz_k20_n_var,
                description="DTLZ7 with disconnected Pareto-optimal regions.",
                factory=lambda n_var, n_obj: _dtlz_factory(DTLZ7Problem, n_var, n_obj),
            ),
        }
    )
    return SPECS


__all__ = ["SPECS", "get_specs"]
