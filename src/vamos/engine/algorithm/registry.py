"""
Algorithm registry.

Maps algorithm names to builder callables so orchestration code avoids
hard-coded conditionals. Builders are expected to accept (config_dict, kernel)
and return an initialized algorithm instance.
"""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Any, Protocol

from vamos.foundation.kernel.backend import KernelBackend
from vamos.foundation.problem.types import ProblemProtocol

from .nsgaii import NSGAII
from .nsgaiii import NSGAIII
from .moead import MOEAD
from .smsemoa import SMSEMOA
from .spea2 import SPEA2
from .ibea import IBEA
from .smpso import SMPSO


class AlgorithmLike(Protocol):
    def run(
        self,
        problem: ProblemProtocol,
        termination: tuple[str, Any],
        seed: int,
        eval_backend: Any | None = None,
        live_viz: Any | None = None,
    ) -> Mapping[str, Any]: ...


AlgorithmBuilder = Callable[[dict, KernelBackend], AlgorithmLike]

ALGORITHMS: Dict[str, AlgorithmBuilder] = {
    "nsgaii": lambda cfg, kernel: NSGAII(cfg, kernel=kernel),
    "nsgaiii": lambda cfg, kernel: NSGAIII(cfg, kernel=kernel),
    "moead": lambda cfg, kernel: MOEAD(cfg, kernel=kernel),
    "smsemoa": lambda cfg, kernel: SMSEMOA(cfg, kernel=kernel),
    "spea2": lambda cfg, kernel: SPEA2(cfg, kernel=kernel),
    "ibea": lambda cfg, kernel: IBEA(cfg, kernel=kernel),
    "smpso": lambda cfg, kernel: SMPSO(cfg, kernel=kernel),
}


def resolve_algorithm(name: str) -> AlgorithmBuilder:
    key = name.lower()
    try:
        return ALGORITHMS[key]
    except KeyError as exc:
        available = ", ".join(sorted(ALGORITHMS))
        raise ValueError(f"Unsupported algorithm '{name}'. Available: {available}") from exc


__all__ = ["ALGORITHMS", "resolve_algorithm", "AlgorithmBuilder", "AlgorithmLike"]
