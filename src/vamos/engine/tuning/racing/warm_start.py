"""
Warm-start evaluator helper for multi-fidelity tuning.

This module provides utilities to easily implement warm-starting between
fidelity levels when tuning multi-objective algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from .tuning_task import EvalContext


@dataclass
class WarmStartEvaluator:
    """
    Helper class for warm-starting algorithms between fidelity levels.

    Wraps your algorithm execution to automatically handle checkpointing
    and continuation between fidelity levels.

    Example usage:

    ```python
    from vamos.engine.tuning.racing import WarmStartEvaluator, EvalContext
    from vamos import optimize, OptimizeConfig, NSGAIIConfig

    def run_algorithm(config: dict, ctx: EvalContext, checkpoint: Any = None):
        # Build algorithm config
        algo_cfg = NSGAIIConfig() \\
            .pop_size(config["pop_size"]) \\
            .crossover("sbx", prob=config["crossover_prob"]) \\
            .fixed()

        opt_cfg = OptimizeConfig(
            problem=ctx.instance.name,
            algorithm=algo_cfg,
            max_evaluations=ctx.budget,
            seed=ctx.seed,
        )

        # If we have a checkpoint, use it as initial population
        if checkpoint is not None:
            opt_cfg = opt_cfg.with_initial_population(checkpoint["X"])

        result = optimize(opt_cfg)

        # Return result and checkpoint for next level
        new_checkpoint = {"X": result.X, "F": result.F}
        return result, new_checkpoint

    # Create evaluator
    evaluator = WarmStartEvaluator(
        run_fn=run_algorithm,
        score_fn=lambda result, ctx: compute_hypervolume(result.F, ref_point),
    )

    # Use in tuner
    best_config, history = tuner.run(evaluator)
    ```
    """

    run_fn: Callable[[Dict[str, Any], EvalContext, Optional[Any]], Tuple[Any, Any]]
    """
    Function that runs the algorithm.

    Args:
        config: Algorithm configuration dict
        ctx: Evaluation context with budget, instance, seed, etc.
        checkpoint: Previous checkpoint (None for first fidelity level)

    Returns:
        Tuple of (result, new_checkpoint)
    """

    score_fn: Callable[[Any, EvalContext], float]
    """
    Function that computes a score from the result.

    Args:
        result: The result returned by run_fn
        ctx: Evaluation context

    Returns:
        Scalar score (higher is better for maximize=True)
    """

    # Optional: track global bounds for normalization
    global_min: Optional[np.ndarray] = None
    global_max: Optional[np.ndarray] = None
    _objectives_seen: bool = field(default=False, init=False)

    def __call__(self, config: Dict[str, Any], ctx: EvalContext) -> Tuple[float, Any]:
        """
        Execute the algorithm and return (score, checkpoint) tuple.

        This method is called by RacingTuner. When fidelity_warm_start=True,
        it passes ctx.checkpoint from the previous fidelity level.
        """
        # Run the algorithm with optional checkpoint
        result, new_checkpoint = self.run_fn(config, ctx, ctx.checkpoint)

        # Update global bounds if result has F (objectives)
        if hasattr(result, "F") and result.F is not None:
            self._update_bounds(result.F)

        # Compute score
        score = float(self.score_fn(result, ctx))

        return score, new_checkpoint

    def _update_bounds(self, F: np.ndarray) -> None:
        """Update global min/max for normalization."""
        if self.global_min is None:
            self.global_min = F.min(axis=0).copy()
            self.global_max = F.max(axis=0).copy()
        else:
            self.global_min = np.minimum(self.global_min, F.min(axis=0))
            self.global_max = np.maximum(self.global_max, F.max(axis=0))
        self._objectives_seen = True

    def normalize(self, F: np.ndarray) -> np.ndarray:
        """
        Normalize objectives using accumulated global bounds.

        Useful for computing normalized hypervolume during tuning.
        """
        if self.global_min is None or self.global_max is None:
            return F

        span = self.global_max - self.global_min
        span = np.where(span < 1e-12, 1.0, span)
        return (F - self.global_min) / span

    def compute_normalized_hv(self, F: np.ndarray, ref_offset: float = 0.05) -> float:
        """
        Compute normalized hypervolume using accumulated global bounds.

        This is useful for problems where ideal/nadir are unknown.
        Supports any number of objectives (uses moocore backend for >2D).
        """
        from vamos.foundation.metrics.hypervolume import hypervolume

        norm_F = self.normalize(F)
        ref = np.ones(F.shape[1]) * (1.0 + ref_offset)
        hv = hypervolume(norm_F, ref)
        max_hv = float(np.prod(ref))

        return float(hv / max_hv) if max_hv > 0 else 0.0


__all__ = ["WarmStartEvaluator"]
