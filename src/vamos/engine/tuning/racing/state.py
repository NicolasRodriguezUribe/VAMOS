from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ConfigState:
    """
    Internal structure to keep track of a single configuration during racing.

    For multi-fidelity warm-starting:
    - `checkpoint`: Stores algorithm state from last evaluation for warm-starting
    - `last_budget`: Budget used in the last evaluation (for cumulative budgets)
    """

    config_id: int
    config: Dict[str, Any]
    alive: bool = True
    # Scores are stored in the same order as the evaluation schedule.
    scores: List[float] = field(default_factory=list)
    # Warm-start support
    checkpoint: Optional[Any] = None
    last_budget: int = 0


@dataclass
class EliteEntry:
    """
    Elite archive entry: configuration and its aggregated score.
    """

    config: Dict[str, Any]
    score: float


__all__ = ["ConfigState", "EliteEntry"]
