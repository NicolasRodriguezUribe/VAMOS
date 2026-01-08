"""
AGE-MOEA: Adaptive Geometry Estimation MOEA.

Reference:
    Panichella, A. (2019). An Adaptive Evolutionary Algorithm based on
    Non-Euclidean Geometry for Many-objective Optimization. GECCO 2019.
"""

from __future__ import annotations

import logging
from typing import Any, Tuple

import numpy as np

from vamos.foundation.kernel.backend import KernelBackend
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.types import ProblemProtocol
from vamos.foundation.eval.backends import EvaluationBackend, SerialEvalBackend
from vamos.engine.algorithm.components.variation.pipeline import VariationPipeline
from vamos.engine.config.variation import resolve_default_variation_config


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def _normalize_objectives(F: np.ndarray) -> np.ndarray:
    """Normalize objectives to [0, 1] range."""
    # ... (unchanged) ...
    f_min = F.min(axis=0)
    f_max = F.max(axis=0)
    diff = f_max - f_min
    diff[diff == 0] = 1.0
    return (F - f_min) / diff


def _compute_geometry_p(F_norm: np.ndarray) -> float:
    # ... (unchanged) ...
    """Estimate geometry parameter p for survival selection."""
    n_points, n_obj = F_norm.shape
    if n_points < 2:
        return 1.0

    # Compute distances from each point to the ideal point (0, 0, ...)
    distances = np.linalg.norm(F_norm, axis=1)

    # Estimate curvature based on distance distribution
    mean_dist = distances.mean()
    if mean_dist < 0.5:
        return 1.0  # Convex-like
    elif mean_dist > 0.7:
        return 0.5  # Concave-like
    else:
        return 1.0 / n_obj  # Mixed


def _age_survival(F: np.ndarray, n_survive: int) -> np.ndarray:
    # ... (unchanged) ...
    """AGE survival selection based on adaptive geometry."""
    n_points = len(F)
    if n_points <= n_survive:
        return np.arange(n_points)

    F_norm = _normalize_objectives(F)
    p = _compute_geometry_p(F_norm)

    # Compute contribution scores using Lp distance
    survivors = []
    remaining = list(range(n_points))

    # Keep extreme points
    for m in range(F.shape[1]):
        best_idx = np.argmin(F[:, m])
        if best_idx not in survivors:
            survivors.append(best_idx)
            remaining.remove(best_idx)

    # Fill rest based on crowding in Lp space
    while len(survivors) < n_survive and remaining:
        max_contrib = -1
        best_idx = remaining[0]

        for idx in remaining:
            # Distance to nearest survivor
            if survivors:
                dists = np.linalg.norm(F_norm[survivors] - F_norm[idx], ord=p, axis=1)
                min_dist = dists.min()
            else:
                min_dist = np.inf

            if min_dist > max_contrib:
                max_contrib = min_dist
                best_idx = idx

        survivors.append(best_idx)
        remaining.remove(best_idx)

    return np.array(survivors)


class AGEMOEA:
    """
    AGE-MOEA: Adaptive Geometry Estimation MOEA.

    Works well on problems with unknown Pareto front shape by
    adapting the geometry estimation during search.
    """

    def __init__(self, config: dict[str, Any], kernel: KernelBackend | None = None):
        self.config = config
        self.kernel = kernel or NumPyKernel()
        self._state = None

    def run(
        self,
        problem: ProblemProtocol,
        termination: Tuple[str, Any],
        seed: int,
        eval_backend: EvaluationBackend | None = None,
        live_viz: Any | None = None,
    ) -> dict[str, Any]:
        """Run AGE-MOEA optimization."""
        rng = np.random.default_rng(seed)
        backend = eval_backend or SerialEvalBackend()

        pop_size = self.config.get("pop_size", 100)
        n_var = problem.n_var
        xl = np.atleast_1d(problem.xl)
        xu = np.atleast_1d(problem.xu)

        term_key, term_val = termination
        max_evals = term_val if term_key == "n_eval" else term_val * pop_size

        # Initialize population
        X = rng.uniform(xl, xu, size=(pop_size, n_var))
        F = backend.evaluate(X, problem).F
        n_eval = pop_size

        # Variation pipeline
        encoding = getattr(problem, "encoding", "real")

        # Extract explicit overrides from config
        explicit_overrides = {}
        if "crossover" in self.config:
            explicit_overrides["crossover"] = self.config["crossover"]
        if "mutation" in self.config:
            explicit_overrides["mutation"] = self.config["mutation"]

        var_cfg = resolve_default_variation_config(encoding, explicit_overrides)

        c_name, c_kwargs = var_cfg.get("crossover", ("sbx", {}))
        m_name, m_kwargs = var_cfg.get("mutation", ("pm", {}))

        variation = VariationPipeline(
            encoding=encoding,
            cross_method=c_name,
            mut_method=m_name,
            cross_params=c_kwargs,
            mut_params=m_kwargs,
            xl=xl,
            xu=xu,
            workspace=None,
        )

        # Setup external archive
        from vamos.archive.bounded_archive import BoundedArchive, BoundedArchiveConfig

        archive = None
        archive_cfg = self.config.get("archive")
        if archive_cfg and archive_cfg.get("size", 0) > 0:
            bac = BoundedArchiveConfig(
                size_cap=int(archive_cfg["size"]),
                archive_type=archive_cfg.get("archive_type", "size_cap"),
                prune_policy=archive_cfg.get("prune_policy", "crowding"),
                epsilon=float(archive_cfg.get("epsilon", 0.01)),
                rng_seed=seed,
                nondominated_only=True,
            )
            archive = BoundedArchive(bac)
            # Add initial population
            archive.add(X, F, n_eval)

        generation = 0
        while n_eval < max_evals:
            # Generate offspring
            parents_idx = rng.integers(0, pop_size, size=pop_size)
            X_off = variation.produce_offspring(X[parents_idx], rng)
            F_off = backend.evaluate(X_off, problem).F
            n_eval += len(X_off)

            # Update external archive with new points
            if archive is not None:
                archive.add(X_off, F_off, n_eval)

            # Combine populations
            X_combined = np.vstack([X, X_off])
            F_combined = np.vstack([F, F_off])

            # AGE survival selection
            survivors = _age_survival(F_combined, pop_size)
            X = X_combined[survivors]
            F = F_combined[survivors]

            generation += 1

        # Determine result based on mode
        result_mode = self.config.get("result_mode", "non_dominated")

        if result_mode == "archive" and archive is not None:
            return {
                "X": archive.X,
                "F": archive.F,
                "n_eval": n_eval,
                "n_gen": generation,
            }

        # Return non-dominated solutions from final population
        ranks, _ = self.kernel.nsga2_ranking(F)
        front_mask = ranks == 0

        return {
            "X": X[front_mask],
            "F": F[front_mask],
            "n_eval": n_eval,
            "n_gen": generation,
        }
