from __future__ import annotations

DEFAULT_ALGORITHM = "nsgaii"

ENABLED_ALGORITHMS = (
    "nsgaii",
    "moead",
    "smsemoa",
    "nsgaiii",
    "spea2",
    "ibea",
    "smpso",
    "agemoea",
    "rvea",
)

OPTIONAL_ALGORITHMS: tuple[str, ...] = ()

__all__ = [
    "DEFAULT_ALGORITHM",
    "ENABLED_ALGORITHMS",
    "OPTIONAL_ALGORITHMS",
]
