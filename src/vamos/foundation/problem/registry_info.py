from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from vamos.foundation.problem.registry.specs import ProblemSpec, get_problem_specs


@dataclass
class ProblemInfo:
    name: str
    description: str
    categories: List[str] = field(default_factory=list)
    default_n_variables: int | None = None
    default_n_objectives: int | None = None
    tags: List[str] = field(default_factory=list)
    encoding: str = "continuous"


def _spec_to_info(spec: ProblemSpec) -> ProblemInfo:
    cats = []
    if spec.encoding:
        cats.append(spec.encoding)
    return ProblemInfo(
        name=spec.key,
        description=spec.description,
        categories=cats,
        default_n_variables=spec.default_n_var,
        default_n_objectives=spec.default_n_obj,
        tags=[spec.label],
        encoding=spec.encoding,
    )


def list_problems() -> List[ProblemInfo]:
    specs = get_problem_specs()
    return [_spec_to_info(spec) for spec in specs.values()]


def get_problem_info(name: str) -> Optional[ProblemInfo]:
    spec = get_problem_specs().get(name)
    if spec is None:
        return None
    return _spec_to_info(spec)


__all__ = ["ProblemInfo", "list_problems", "get_problem_info"]
