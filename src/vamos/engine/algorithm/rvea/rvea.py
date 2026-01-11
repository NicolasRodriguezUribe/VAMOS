"""
RVEA: Reference Vector-guided Evolutionary Algorithm.

Reference:
    Cheng, R., Jin, Y., Olhofer, M., & Sendhoff, B. (2016).
    A Reference Vector Guided Evolutionary Algorithm for Many-objective Optimization.
    IEEE TEVC.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Tuple

import numpy as np

from vamos.engine.algorithm.components.population import initialize_population, resolve_bounds
from vamos.engine.algorithm.components.variation.pipeline import VariationPipeline
from vamos.engine.config.variation import resolve_default_variation_config
from vamos.foundation.eval.backends import EvaluationBackend, SerialEvalBackend
from vamos.foundation.kernel.backend import KernelBackend
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.types import ProblemProtocol


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def _generate_reference_vectors(n_obj: int, n_partitions: int = 12) -> np.ndarray:
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

    ref_dirs = _das_dennis(n_partitions, n_obj)
    return ref_dirs


def _calc_V(ref_dirs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(ref_dirs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return ref_dirs / norms


def _calc_gamma(V: np.ndarray) -> np.ndarray:
    cosine = V @ V.T
    gamma = np.arccos((-np.sort(-1.0 * cosine, axis=1))[:, 1])
    gamma = np.maximum(gamma, 1e-64)
    return gamma


def _apd_survival(
    F: np.ndarray,
    V: np.ndarray,
    gamma: np.ndarray,
    ideal: np.ndarray,
    n_survive: int,
    n_gen: int,
    n_max_gen: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    n_obj = F.shape[1]
    ideal = np.minimum(F.min(axis=0), ideal)

    F_shift = F - ideal
    dist_to_ideal = np.linalg.norm(F_shift, axis=1)
    dist_to_ideal[dist_to_ideal < 1e-64] = 1e-64
    F_prime = F_shift / dist_to_ideal[:, None]

    cos_theta = np.clip(F_prime @ V.T, -1.0, 1.0)
    acute_angle = np.arccos(cos_theta)
    niches = acute_angle.argmin(axis=1)

    survivors = []
    for k in range(len(V)):
        assigned = np.where(niches == k)[0]
        if assigned.size == 0:
            continue
        theta = acute_angle[assigned, k]
        M = float(n_obj) if n_obj > 2 else 1.0
        penalty = M * ((n_gen / n_max_gen) ** alpha) * (theta / gamma[k])
        apd = dist_to_ideal[assigned] * (1.0 + penalty)
        survivor = assigned[int(np.argmin(apd))]
        survivors.append(survivor)

    survivors = np.asarray(survivors, dtype=int)
    nadir = F[survivors].max(axis=0) if survivors.size else None
    return survivors, ideal, nadir


class RVEA:
    """
    RVEA: Reference Vector-guided Evolutionary Algorithm.

    Implements angle-penalized distance (APD) survival and periodic
    reference-vector adaptation.
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
        eval_strategy: EvaluationBackend | None = None,
        live_viz: Any | None = None,
    ) -> dict[str, Any]:
        """Run RVEA optimization."""
        rng = np.random.default_rng(seed)
        backend = eval_strategy or SerialEvalBackend()

        pop_size = int(self.config.get("pop_size", 100))
        n_obj = int(problem.n_obj)
        n_partitions = int(self.config.get("n_partitions", 12))
        alpha = float(self.config.get("alpha", 2.0))
        adapt_freq = self.config.get("adapt_freq", 0.1)

        term_key, term_val = termination
        if term_key == "n_gen":
            max_gen = max(1, int(term_val))
            max_evals = max_gen * pop_size
        else:
            max_evals = int(term_val)
            max_gen = max(1, int(math.ceil((max_evals - pop_size) / pop_size)))

        ref_dirs = _generate_reference_vectors(n_obj, n_partitions)
        if ref_dirs.shape[0] != pop_size:
            raise ValueError(
                f"RVEA requires pop_size == #ref_dirs for partitions={n_partitions} (pop_size={pop_size}, ref_dirs={ref_dirs.shape[0]})."
            )
        V = _calc_V(ref_dirs)
        gamma = _calc_gamma(V)

        encoding = getattr(problem, "encoding", "real")
        xl, xu = resolve_bounds(problem, encoding)
        X = initialize_population(pop_size, problem.n_var, xl, xu, encoding, rng, problem, self.config.get("initializer"))
        F = np.asarray(backend.evaluate(X, problem).F, dtype=float)
        n_eval = X.shape[0]

        explicit_overrides = {}
        if "crossover" in self.config:
            explicit_overrides["crossover"] = self.config["crossover"]
        if "mutation" in self.config:
            explicit_overrides["mutation"] = self.config["mutation"]
        if "repair" in self.config:
            explicit_overrides["repair"] = self.config["repair"]

        var_cfg = resolve_default_variation_config(encoding, explicit_overrides)
        c_name, c_kwargs = var_cfg.get("crossover", ("sbx", {}))
        m_name, m_kwargs = var_cfg.get("mutation", ("pm", {}))
        repair_cfg = var_cfg.get("repair")

        variation = VariationPipeline(
            encoding=encoding,
            cross_method=c_name,
            mut_method=m_name,
            cross_params=c_kwargs,
            mut_params=m_kwargs,
            xl=xl,
            xu=xu,
            workspace=None,
            repair_cfg=repair_cfg,
            problem=problem,
        )

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
            archive.add(X, F, n_eval)

        ideal = np.full(n_obj, np.inf)
        nadir = None
        adapt_interval = None
        if adapt_freq is not None:
            adapt_interval = max(1, int(math.ceil(max_gen * float(adapt_freq))))

        generation = 1
        while n_eval < max_evals:
            parents_idx = rng.integers(0, len(X), size=pop_size)
            X_off = variation.produce_offspring(X[parents_idx], rng)
            F_off = np.asarray(backend.evaluate(X_off, problem).F, dtype=float)
            n_eval += X_off.shape[0]

            if archive is not None:
                archive.add(X_off, F_off, n_eval)

            X_combined = np.vstack([X, X_off])
            F_combined = np.vstack([F, F_off])

            survivors, ideal, nadir = _apd_survival(
                F_combined,
                V,
                gamma,
                ideal,
                pop_size,
                generation,
                max_gen,
                alpha,
            )
            if survivors.size == 0:
                break
            X = X_combined[survivors]
            F = F_combined[survivors]

            if adapt_interval is not None and generation % adapt_interval == 0 and nadir is not None:
                scale = np.maximum(nadir - ideal, 1e-64)
                V = _calc_V(_calc_V(ref_dirs) * scale)
                gamma = _calc_gamma(V)

            generation += 1

        result_mode = self.config.get("result_mode", "non_dominated")
        if result_mode == "archive" and archive is not None:
            return {"X": archive.X, "F": archive.F, "n_eval": n_eval, "n_gen": generation}

        ranks, _ = self.kernel.nsga2_ranking(F)
        front_mask = ranks == 0
        return {"X": X[front_mask], "F": F[front_mask], "n_eval": n_eval, "n_gen": generation}
