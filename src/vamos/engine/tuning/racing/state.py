from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


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
    checkpoint_map: Dict[Tuple[int, int], Any] = field(default_factory=dict)
    last_budget_map: Dict[Tuple[int, int], int] = field(default_factory=dict)
    # Multi-fidelity score tracking
    fidelity_scores: Dict[int, List[float]] = field(default_factory=dict)


@dataclass
class EliteEntry:
    """
    Elite archive entry: configuration and its aggregated score.
    """

    config: Dict[str, Any]
    score: float


__all__ = ["ConfigState", "EliteEntry"]
