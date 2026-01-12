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

from vamos.engine.algorithm.components.population import initialize_population, resolve_bounds
from vamos.engine.algorithm.components.variation.pipeline import VariationPipeline
from vamos.engine.config.variation import resolve_default_variation_config
from vamos.foundation.encoding import normalize_encoding
from vamos.foundation.eval.backends import EvaluationBackend, SerialEvalBackend
from vamos.foundation.kernel.backend import KernelBackend
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.types import ProblemProtocol


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def _point_to_line_distance(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    d = np.zeros(P.shape[0], dtype=float)
    ba = B - A
    denom = np.dot(ba, ba)
    if denom == 0.0:
        return d
    for i in range(P.shape[0]):
        pa = P[i] - A
        t = np.dot(pa, ba) / denom
        d[i] = np.sum((pa - t * ba) ** 2)
    return d


def _find_corner_solutions(front: np.ndarray) -> np.ndarray:
    m, n = front.shape
    if m <= n:
        return np.arange(m)
    W = 1e-6 + np.eye(n)
    indexes = np.zeros(n, dtype=int)
    selected = np.zeros(m, dtype=bool)
    for i in range(n):
        dists = _point_to_line_distance(front, np.zeros(n), W[i, :])
        dists[selected] = np.inf
        idx = int(np.argmin(dists))
        indexes[i] = idx
        selected[idx] = True
    return indexes


def _normalize_front(front: np.ndarray, extreme: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(extreme) != len(np.unique(extreme, axis=0)):
        normalization = np.max(front, axis=0)
        normalization[normalization == 0.0] = 1.0
        return front / normalization, normalization

    try:
        hyperplane = np.linalg.solve(front[extreme], np.ones(front.shape[1]))
        if np.any(~np.isfinite(hyperplane)) or np.any(hyperplane <= 0):
            normalization = np.max(front, axis=0)
        else:
            normalization = 1.0 / hyperplane
            if np.any(~np.isfinite(normalization)):
                normalization = np.max(front, axis=0)
    except np.linalg.LinAlgError:
        normalization = np.max(front, axis=0)

    normalization[normalization == 0.0] = 1.0
    return front / normalization, normalization


def _pairwise_distances(front: np.ndarray, p: float) -> np.ndarray:
    m = front.shape[0]
    distances = np.zeros((m, m), dtype=float)
    for i in range(m):
        distances[i] = np.sum(np.abs(front[i] - front) ** p, axis=1) ** (1.0 / p)
    return distances


def _minkowski_distances(A: np.ndarray, B: np.ndarray, p: float) -> np.ndarray:
    m1 = A.shape[0]
    m2 = B.shape[0]
    distances = np.zeros((m1, m2), dtype=float)
    for i in range(m1):
        for j in range(m2):
            distances[i][j] = np.sum(np.abs(A[i] - B[j]) ** p) ** (1.0 / p)
    return distances


def _compute_geometry(front: np.ndarray, extreme: np.ndarray, n_obj: int) -> float:
    d = _point_to_line_distance(front, np.zeros(n_obj), np.ones(n_obj))
    d[extreme] = np.inf
    index = int(np.argmin(d))
    mean_val = np.mean(front[index, :])
    if mean_val <= 0.0:
        return 1.0
    p = np.log(n_obj) / np.log(1.0 / mean_val)
    if np.isnan(p) or p <= 0.1:
        p = 1.0
    elif p > 20.0:
        p = 20.0
    return float(p)


def _survival_score(front: np.ndarray, ideal_point: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    front = np.round(front, 12, out=front.copy())
    m, n = front.shape
    crowd_dist = np.zeros(m, dtype=float)

    if m < n:
        p = 1.0
        normalization = np.max(front, axis=0)
        normalization[normalization == 0.0] = 1.0
        return crowd_dist, p, normalization

    front = front - ideal_point
    extreme = _find_corner_solutions(front)
    front, normalization = _normalize_front(front, extreme)

    crowd_dist[extreme] = np.inf
    selected = np.full(m, False)
    selected[extreme] = True

    p = _compute_geometry(front, extreme, n)
    nn = np.linalg.norm(front, ord=p, axis=1)
    nn[nn < 1e-8] = 1.0

    distances = _pairwise_distances(front, p)
    distances[distances < 1e-8] = 1e-8
    distances = distances / nn[:, None]

    neighbors = 2
    remaining = list(np.arange(m)[~selected])
    for _ in range(m - np.sum(selected)):
        selected_idx = np.arange(selected.shape[0])[selected]
        mg = np.meshgrid(selected_idx, remaining, copy=False, sparse=False)
        D_mg = distances[tuple(mg)]

        if D_mg.shape[1] > 1:
            maxim = np.argpartition(D_mg, neighbors - 1, axis=1)[:, :neighbors]
            tmp = np.sum(np.take_along_axis(D_mg, maxim, axis=1), axis=1)
            index = int(np.argmax(tmp))
            d = tmp[index]
        else:
            index = int(np.argmax(D_mg[:, 0]))
            d = D_mg[index, 0]

        best = remaining.pop(index)
        selected[best] = True
        crowd_dist[best] = d

    return crowd_dist, p, normalization


def _age_survival(F: np.ndarray, n_survive: int, kernel: KernelBackend) -> np.ndarray:
    ranks, _ = kernel.nsga2_ranking(F)
    max_rank = int(ranks.max()) if ranks.size else 0

    fronts = []
    ranked = 0
    last_rank = 0
    for r in range(max_rank + 1):
        front = np.where(ranks == r)[0]
        fronts.append(front)
        if ranked + front.size >= n_survive:
            last_rank = r
            break
        ranked += front.size

    selected = ranks < last_rank
    crowd_dist = np.zeros(F.shape[0], dtype=float)

    front0 = F[ranks == 0, :]
    ideal_point = np.min(front0, axis=0)
    crowd_dist[ranks == 0], p, normalization = _survival_score(front0, ideal_point)

    for r in range(1, last_rank):
        front_idx = fronts[r]
        if front_idx.size == 0:
            continue
        front = F[front_idx] / normalization
        dist = _minkowski_distances(front, ideal_point[None, :], p).squeeze()
        dist = np.where(dist < 1e-12, 1e-12, dist)
        crowd_dist[front_idx] = 1.0 / dist

    last = fronts[last_rank]
    if last.size > 0:
        order = np.argsort(crowd_dist[last])[::-1]
        remaining = n_survive - int(np.sum(selected))
        selected[last[order[:remaining]]] = True

    return np.flatnonzero(selected)


class AGEMOEA:
    """
    AGE-MOEA: Adaptive Geometry Estimation MOEA.

    Implements adaptive geometry estimation for survival selection as in the
    original AGE-MOEA paper.
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
        """Run AGE-MOEA optimization."""
        rng = np.random.default_rng(seed)
        backend = eval_strategy or SerialEvalBackend()

        pop_size = int(self.config.get("pop_size", 100))
        term_key, term_val = termination
        max_evals = term_val if term_key == "n_eval" else int(term_val) * pop_size

        encoding = normalize_encoding(getattr(problem, "encoding", "real"))
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

        generation = 0
        while n_eval < max_evals:
            ranks, crowding = self.kernel.nsga2_ranking(F)
            n_parents = 2 * (pop_size // 2)
            parents_idx = self.kernel.tournament_selection(
                ranks=ranks,
                crowding=crowding,
                pressure=2,
                rng=rng,
                n_parents=n_parents,
            )

            X_off = variation.produce_offspring(X[parents_idx], rng)
            F_off = np.asarray(backend.evaluate(X_off, problem).F, dtype=float)
            n_eval += X_off.shape[0]

            if archive is not None:
                archive.add(X_off, F_off, n_eval)

            X_combined = np.vstack([X, X_off])
            F_combined = np.vstack([F, F_off])

            survivors = _age_survival(F_combined, pop_size, self.kernel)
            X = X_combined[survivors]
            F = F_combined[survivors]

            generation += 1

        result_mode = self.config.get("result_mode", "non_dominated")
        if result_mode == "archive" and archive is not None:
            return {"X": archive.X, "F": archive.F, "n_eval": n_eval, "n_gen": generation}

        ranks, _ = self.kernel.nsga2_ranking(F)
        front_mask = ranks == 0
        return {"X": X[front_mask], "F": F[front_mask], "n_eval": n_eval, "n_gen": generation}
