from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ConfigState:
    """
    Internal structure to keep track of a single configuration during racing.
    """

    config_id: int
    config: Dict[str, Any]
    alive: bool = True
    # Scores are stored in the same order as the evaluation schedule.
    scores: List[float] = field(default_factory=list)


@dataclass
class EliteEntry:
    """
    Elite archive entry: configuration and its aggregated score.
    """

    config: Dict[str, Any]
    score: float


__all__ = ["ConfigState", "EliteEntry"]
