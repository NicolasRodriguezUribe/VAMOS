from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SpecDefaults:
    spec: dict[str, Any]
    problem_overrides: dict[str, Any]
    experiment_defaults: dict[str, Any]
    nsgaii_defaults: dict[str, Any]
    moead_defaults: dict[str, Any]
    smsemoa_defaults: dict[str, Any]
    nsgaiii_defaults: dict[str, Any]
