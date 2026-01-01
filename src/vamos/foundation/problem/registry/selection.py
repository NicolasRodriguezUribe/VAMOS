from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .specs import ProblemSpec, PROBLEM_SPECS
from typing import cast

from ..types import ProblemProtocol


@dataclass(frozen=True)
class ProblemSelection:
    """Concrete problem instance choice with resolved dimensions."""

    spec: ProblemSpec
    n_var: int
    n_obj: int

    def instantiate(self) -> ProblemProtocol:
        """Create a new problem instance with the resolved dimensions."""
        return cast(ProblemProtocol, self.spec.factory(self.n_var, self.n_obj))


def make_problem_selection(key: str, *, n_var: Optional[int] = None, n_obj: Optional[int] = None) -> ProblemSelection:
    try:
        spec = PROBLEM_SPECS[key]
    except KeyError as exc:
        raise KeyError(f"Unknown problem '{key}'.") from exc

    actual_n_var, actual_n_obj = spec.resolve_dimensions(n_var=n_var, n_obj=n_obj)
    return ProblemSelection(spec=spec, n_var=actual_n_var, n_obj=actual_n_obj)


__all__ = ["ProblemSelection", "make_problem_selection"]
