from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .param_space import ParamSpace


@dataclass
class Instance:
    """
    Represents a single problem instance used during tuning.
    """

    name: str
    n_var: int
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalContext:
    """
    Information passed to the evaluation function.
    
    For multi-fidelity warm-starting:
    - `fidelity_level`: Current fidelity level (0-indexed)
    - `previous_budget`: Budget used in previous fidelity level (None if first level)
    - `checkpoint`: Algorithm state from previous fidelity level (None if first level)
    
    The eval_fn can use `checkpoint` to warm-start the algorithm instead of
    starting from scratch.
    """

    instance: Instance
    seed: int
    budget: int
    fidelity_level: int = 0
    previous_budget: Optional[int] = None
    checkpoint: Optional[Any] = None


@dataclass
class TuningTask:
    """
    Describes a tuning task for VAMOS:
      - parameter space
      - set of instances
      - seeds
      - budget per run
      - how to aggregate scores across (instance, seed).
    """

    name: str
    param_space: "ParamSpace"
    instances: Sequence[Instance]
    seeds: Sequence[int]
    budget_per_run: int
    maximize: bool = True
    aggregator: Callable[[List[float]], float] = np.mean

    def eval_config(
        self,
        config: Dict[str, Any],
        eval_fn: Callable[[Dict[str, Any], EvalContext], float],
    ) -> float:
        """
        Evaluate a configuration across all (instance, seed) combinations
        and aggregate the scores into a single scalar.
        """
        scores: List[float] = []
        for inst in self.instances:
            for seed in self.seeds:
                ctx = EvalContext(instance=inst, seed=seed, budget=self.budget_per_run)
                score = float(eval_fn(config, ctx))
                scores.append(score)
        if not scores:
            raise RuntimeError("No scores computed for configuration")
        return float(self.aggregator(scores))


__all__ = ["TuningTask", "EvalContext", "Instance"]
