"""
RVEA: Reference Vector-guided Evolutionary Algorithm.

Reference:
    Cheng, R., Jin, Y., Olhofer, M., & Sendhoff, B. (2016).
    A Reference Vector Guided Evolutionary Algorithm for Many-objective Optimization.
    IEEE TEVC.
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


def _generate_reference_vectors(n_obj: int, n_partitions: int = 12) -> np.ndarray:
    # ... (unchanged) ...
    """Generate uniformly distributed reference vectors using Das-Dennis."""
    from itertools import combinations

    def _das_dennis(n_partitions: int, n_obj: int) -> np.ndarray:
        if n_obj == 1:
            return np.array([[1.0]])

        vectors = []
        for indices in combinations(range(n_partitions + n_obj - 1), n_obj - 1):
            prev = -1
            vec = []
            for idx in indices:
                vec.append(idx - prev - 1)
                prev = idx
            vec.append(n_partitions + n_obj - 2 - prev)
            vectors.append(vec)

        vectors = np.array(vectors, dtype=float) / n_partitions
        return vectors

    V = _das_dennis(n_partitions, n_obj)
    # Normalize
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return V / norms


def _angle_penalized_distance(F: np.ndarray, V: np.ndarray, gamma: float) -> np.ndarray:
    # ... (unchanged) ...
    """Compute angle-penalized distance (APD) for each solution to each reference vector."""
    n_points = len(F)
    n_vectors = len(V)

    # Normalize objectives to translate
    f_min = F.min(axis=0)
    F_trans = F - f_min + 1e-6

    # Compute cosine similarity
    F_norm = F_trans / np.linalg.norm(F_trans, axis=1, keepdims=True)

    # Distance to each reference vector
    apd = np.zeros((n_points, n_vectors))
    for j in range(n_vectors):
        cos_theta = F_norm @ V[j]
        cos_theta = np.clip(cos_theta, -1, 1)
        theta = np.arccos(cos_theta)

        # APD = (1 + gamma * theta) * ||F||
        d1 = np.linalg.norm(F_trans, axis=1)
        penalty = 1 + gamma * theta
        apd[:, j] = penalty * d1

    return apd


def _rvea_survival(F: np.ndarray, V: np.ndarray, n_survive: int, gen: int, max_gen: int) -> np.ndarray:
    # ... (unchanged) ...
    """RVEA survival selection using adaptive reference vectors."""
    n_points = len(F)
    n_vectors = len(V)

    if n_points <= n_survive:
        return np.arange(n_points)

    # Adaptive penalty parameter
    gamma = (gen / max_gen) ** 2

    # Compute APD
    apd = _angle_penalized_distance(F, V, gamma)

    # Associate each solution to closest reference vector
    associations = np.argmin(apd, axis=1)

    # Select best solution for each reference vector
    survivors = []
    for j in range(n_vectors):
        mask = associations == j
        if mask.sum() > 0:
            candidates = np.where(mask)[0]
            best = candidates[np.argmin(apd[candidates, j])]
            survivors.append(best)

    # If not enough, fill with remaining best
    survivors = list(set(survivors))
    remaining = [i for i in range(n_points) if i not in survivors]

    while len(survivors) < n_survive and remaining:
        # Add solution with best overall APD
        best_idx = remaining[np.argmin(apd[remaining].min(axis=1))]
        survivors.append(best_idx)
        remaining.remove(best_idx)

    return np.array(survivors[:n_survive])


class RVEA:
    """
    RVEA: Reference Vector-guided Evolutionary Algorithm.

    Designed for many-objective optimization (>3 objectives) using
    adaptive reference vectors.
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
        """Run RVEA optimization."""
        rng = np.random.default_rng(seed)
        backend = eval_backend or SerialEvalBackend()

        pop_size = self.config.get("pop_size", 100)
        n_var = problem.n_var
        n_obj = problem.n_obj
        xl = np.atleast_1d(problem.xl)
        xu = np.atleast_1d(problem.xu)

        term_key, term_val = termination
        max_evals = term_val if term_key == "n_eval" else term_val * pop_size
        max_gen = max_evals // pop_size

        # Generate reference vectors
        n_partitions = self.config.get("n_partitions", 12)
        V = _generate_reference_vectors(n_obj, n_partitions)

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
            parents_idx = rng.integers(0, len(X), size=pop_size)
            X_off = variation.produce_offspring(X[parents_idx], rng)
            F_off = backend.evaluate(X_off, problem).F
            n_eval += len(X_off)

            # Update external archive with new points
            if archive is not None:
                archive.add(X_off, F_off, n_eval)

            # Combine populations
            X_combined = np.vstack([X, X_off])
            F_combined = np.vstack([F, F_off])

            # RVEA survival selection
            survivors = _rvea_survival(F_combined, V, pop_size, generation, max_gen)
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
