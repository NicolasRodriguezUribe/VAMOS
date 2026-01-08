"""
Example: Custom Algorithm Plugin for VAMOS

This example demonstrates how to create and register a custom optimization algorithm.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from vamos.engine.algorithm.registry import ALGORITHMS, AlgorithmLike
from vamos.foundation.kernel.backend import KernelBackend
from vamos.foundation.problem.types import ProblemProtocol


class RandomSearchAlgorithm:
    """
    A simple random search algorithm for demonstration purposes.
    
    This is not a competitive algorithm - it's just an example of the interface.
    """

    def __init__(self, config: dict, kernel: KernelBackend) -> None:
        self.config = config
        self.kernel = kernel
        self.pop_size = config.get("pop_size", 100)

    def run(
        self,
        problem: ProblemProtocol,
        termination: tuple[str, Any],
        seed: int,
        eval_backend: Any | None = None,
        live_viz: Any | None = None,
    ) -> Mapping[str, Any]:
        """Run random search optimization."""
        rng = np.random.default_rng(seed)
        
        # Parse termination
        _, max_evals = termination
        if isinstance(max_evals, dict):
            max_evals = max_evals.get("max_evaluations", 1000)
        
        # Initialize
        xl, xu = problem.xl, problem.xu
        best_X = None
        best_F = None
        n_eval = 0
        
        # Notify start
        if live_viz:
            live_viz.on_start(problem=problem, algorithm=self, config=self.config)
        
        generation = 0
        while n_eval < max_evals:
            # Generate random population
            X = rng.uniform(xl, xu, size=(self.pop_size, problem.n_var))
            F = problem.evaluate(X)
            n_eval += self.pop_size
            
            # Keep non-dominated
            if best_F is None:
                best_X, best_F = X, F
            else:
                combined_X = np.vstack([best_X, X])
                combined_F = np.vstack([best_F, F])
                # Simple: just keep best by first objective
                idx = np.argsort(combined_F[:, 0])[:self.pop_size]
                best_X = combined_X[idx]
                best_F = combined_F[idx]
            
            generation += 1
            if live_viz:
                live_viz.on_generation(generation, F=best_F)
        
        if live_viz:
            live_viz.on_end(final_F=best_F)
        
        return {
            "X": best_X,
            "F": best_F,
            "evaluations": n_eval,
        }


# Register the algorithm
@ALGORITHMS.register("random_search")
def build_random_search(cfg: dict, kernel: KernelBackend) -> AlgorithmLike:
    """Factory function for RandomSearchAlgorithm."""
    return RandomSearchAlgorithm(cfg, kernel)


if __name__ == "__main__":
    # Demo usage
    from vamos.experiment.builder import study

    result = study("zdt1").using("random_search").evaluations(1000).run()
    print(f"Found {result.F.shape[0]} solutions")
    print(f"Best f1: {result.F[:, 0].min():.4f}")
