"""
Algorithm registry.

Maps algorithm names to builder callables so orchestration code avoids
hard-coded conditionals. Builders are expected to accept (config_dict, kernel)
and return an initialized algorithm instance.
"""
from __future__ import annotations

from typing import Callable, Dict

from .nsgaii import NSGAII
from .nsga3 import NSGAIII
from .moead import MOEAD
from .smsemoa import SMSEMOA

AlgorithmBuilder = Callable[[dict, object], object]

ALGORITHMS: Dict[str, AlgorithmBuilder] = {
    "nsgaii": lambda cfg, kernel: NSGAII(cfg, kernel=kernel),
    "nsga3": lambda cfg, kernel: NSGAIII(cfg, kernel=kernel),
    "moead": lambda cfg, kernel: MOEAD(cfg, kernel=kernel),
    "smsemoa": lambda cfg, kernel: SMSEMOA(cfg, kernel=kernel),
}


def resolve_algorithm(name: str) -> AlgorithmBuilder:
    key = name.lower()
    try:
        return ALGORITHMS[key]
    except KeyError as exc:
        available = ", ".join(sorted(ALGORITHMS))
        raise ValueError(f"Unsupported algorithm '{name}'. Available: {available}") from exc


__all__ = ["ALGORITHMS", "resolve_algorithm", "AlgorithmBuilder"]
