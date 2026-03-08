"""
AGE-MOEA: Adaptive Geometry Estimation MOEA.

Reference:
    Panichella, A. (2019). An Adaptive Evolutionary Algorithm based on
    Non-Euclidean Geometry for Many-objective Optimization. GECCO 2019.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vamos.engine.archive.bounded_archive import BoundedArchive

import numpy as np

from vamos.engine.algorithm.components.population import initialize_population, resolve_bounds
from vamos.engine.algorithm.components.variation.helpers import (
    ensure_supported_operator_names,
    ensure_supported_repair_name,
)
from vamos.engine.algorithm.components.variation.pipeline import VariationPipeline
from vamos.engine.config.variation import (
    ensure_operator_tuple,
    resolve_default_variation_config,
)
from vamos.foundation.encoding import normalize_encoding
from vamos.foundation.eval.backends import EvaluationBackend, SerialEvalBackend
from vamos.foundation.kernel import default_kernel
from vamos.foundation.kernel.backend import KernelBackend
from vamos.foundation.problem.types import ProblemProtocol

from .state import AGEMOEAState, build_agemoea_result


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def _point_to_line_distance(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    ba = B - A
    denom = np.dot(ba, ba)
    if denom == 0.0:
        return np.zeros(P.shape[0], dtype=float)
    pa = P - A
    t = (pa @ ba) / denom
    residual = pa - t[:, None] * ba
    return np.sum(residual * residual, axis=1)


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
    diff = np.abs(front[:, None, :] - front[None, :, :])
    return np.sum(diff**p, axis=2) ** (1.0 / p)


def _minkowski_distances(A: np.ndarray, B: np.ndarray, p: float) -> np.ndarray:
    diff = np.abs(A[:, None, :] - B[None, :, :])
    return np.sum(diff**p, axis=2) ** (1.0 / p)


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

    remaining = np.flatnonzero(~selected)
    selected_idx = np.flatnonzero(selected)
    if remaining.size == 0:
        return crowd_dist, p, normalization

    D_init = distances[np.ix_(remaining, selected_idx)]
    if D_init.shape[1] > 1:
        nearest = np.partition(D_init, kth=1, axis=1)[:, :2]
        best1 = nearest[:, 0].copy()
        best2 = nearest[:, 1].copy()
        scores = best1 + best2
    else:
        best1 = D_init[:, 0].copy()
        best2 = np.zeros_like(best1)
        scores = best1.copy()

    selected_count = selected_idx.size
    while remaining.size > 0:
        index = int(np.argmax(scores))
        best = int(remaining[index])
        d = float(scores[index])
        selected[best] = True
        crowd_dist[best] = d

        remaining = np.delete(remaining, index)
        best1 = np.delete(best1, index)
        best2 = np.delete(best2, index)
        scores = np.delete(scores, index)
        selected_count += 1
        if remaining.size == 0:
            break

        new_dist = distances[remaining, best]
        if selected_count == 2:
            lo = np.minimum(best1, new_dist)
            hi = np.maximum(best1, new_dist)
            best1 = lo
            best2 = hi
            scores = best1 + best2
            continue

        better_first = new_dist < best1
        best2 = np.where(better_first, best1, best2)
        best1 = np.where(better_first, new_dist, best1)
        better_second = (~better_first) & (new_dist < best2)
        best2 = np.where(better_second, new_dist, best2)
        scores = best1 + best2

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


def _build_variation(config: dict[str, Any], encoding: Any, xl: Any, xu: Any, problem: ProblemProtocol) -> VariationPipeline:
    explicit_overrides: dict[str, Any] = {}
    if "crossover" in config:
        explicit_overrides["crossover"] = config["crossover"]
    if "mutation" in config:
        explicit_overrides["mutation"] = config["mutation"]
    if "repair" in config and config["repair"] != "auto":
        explicit_overrides["repair"] = config["repair"]

    var_cfg = resolve_default_variation_config(encoding, explicit_overrides)
    c_name, c_kwargs = ensure_operator_tuple(var_cfg.get("crossover", ("sbx", {})), key="crossover")
    m_name, m_kwargs = ensure_operator_tuple(var_cfg.get("mutation", ("polynomial", {})), key="mutation")
    repair_cfg: tuple[str, dict[str, Any]] | str = "auto"
    cross_name, mut_name = ensure_supported_operator_names(encoding, c_name, m_name)
    if "repair" in var_cfg:
        repair_name, repair_params = ensure_operator_tuple(var_cfg["repair"], key="repair")
        repair_cfg = (ensure_supported_repair_name(encoding, repair_name), repair_params)

    return VariationPipeline(
        encoding=encoding,
        cross_method=cross_name,
        mut_method=mut_name,
        cross_params=c_kwargs,
        mut_params=m_kwargs,
        xl=xl,
        xu=xu,
        workspace=None,
        repair_cfg=repair_cfg,
        problem=problem,
    )


def _build_archive(config: dict[str, Any], _seed: int) -> BoundedArchive | None:
    from vamos.engine.archive import ExternalArchiveConfig
    from vamos.engine.archive.bounded_archive import BoundedArchive, BoundedArchiveConfig

    ext_cfg = config.get("external_archive")
    if ext_cfg is None:
        return None
    if isinstance(ext_cfg, dict):
        ext_cfg = ExternalArchiveConfig(**ext_cfg)
    if ext_cfg.capacity is None or ext_cfg.capacity <= 0:
        return None
    bac = BoundedArchiveConfig(
        size_cap=ext_cfg.capacity,
        prune_policy=ext_cfg.pruning,
    )
    return BoundedArchive(bac)


class AGEMOEA:
    """AGE-MOEA: Adaptive Geometry Estimation MOEA.

    Implements adaptive geometry estimation for survival selection as in the
    original AGE-MOEA paper.

    Parameters
    ----------
    config : dict
        Algorithm configuration.
    kernel : KernelBackend, optional
        Backend for vectorized operations.

    Examples
    --------
    Batch mode:

    >>> algo = AGEMOEA(config, kernel)
    >>> result = algo.run(problem, ("max_evaluations", 10000), seed=42)

    Ask/tell interface:

    >>> algo = AGEMOEA(config, kernel)
    >>> algo.initialize(problem, ("max_evaluations", 10000), seed=42)
    >>> while not algo.should_terminate():
    ...     X = algo.ask()
    ...     F = evaluate(X)
    ...     algo.tell(F)
    >>> result = algo.result()
    """

    def __init__(self, config: dict[str, Any], kernel: KernelBackend | None = None):
        self.cfg = config
        self.kernel = kernel or default_kernel()
        self._st: AGEMOEAState | None = None

    def _refresh_selection_metrics(self, st: AGEMOEAState) -> None:
        ranks, crowding = self.kernel.nsga2_ranking(st.F)
        st.selection_ranks = np.asarray(ranks, dtype=int)
        st.selection_crowding = np.asarray(crowding, dtype=float)

    # -------------------------------------------------------------------------
    # Main run method (batch mode)
    # -------------------------------------------------------------------------

    def run(
        self,
        problem: ProblemProtocol,
        termination: tuple[str, Any] = ("max_evaluations", 25000),
        seed: int = 0,
        eval_strategy: EvaluationBackend | None = None,
        live_viz: Any | None = None,
    ) -> dict[str, Any]:
        """Run AGE-MOEA optimization."""
        self.initialize(problem, termination, seed, eval_strategy)
        backend = eval_strategy or SerialEvalBackend()

        assert self._st is not None
        while not self.should_terminate():
            X_off = self.ask()
            F_off = np.asarray(backend.evaluate(X_off, problem).F, dtype=float)
            self.tell(F_off)

        return self.result()

    # -------------------------------------------------------------------------
    # Ask/Tell Interface
    # -------------------------------------------------------------------------

    def initialize(
        self,
        problem: ProblemProtocol,
        termination: tuple[str, Any] = ("max_evaluations", 25000),
        seed: int = 0,
        eval_strategy: EvaluationBackend | None = None,
    ) -> None:
        """Initialize algorithm state for ask/tell loop.

        Parameters
        ----------
        problem : ProblemProtocol
            Problem to optimize.
        termination : tuple
            Termination criterion, e.g., ``("max_evaluations", 10000)``.
        seed : int
            Random seed for reproducibility.
        eval_strategy : EvaluationBackend, optional
            Evaluation backend for the initial population.
        """
        rng = np.random.default_rng(seed)
        backend = eval_strategy or SerialEvalBackend()

        pop_size = int(self.cfg.get("pop_size", 100))
        term_key, term_val = termination
        if term_key == "max_evaluations":
            max_evals = int(term_val)
        elif term_key == "n_gen":
            max_evals = int(term_val) * pop_size
        else:
            raise ValueError("Unsupported termination criterion for AGE-MOEA.")

        encoding = normalize_encoding(getattr(problem, "encoding", "real"))
        xl, xu = resolve_bounds(problem, encoding)
        X = initialize_population(pop_size, problem.n_var, xl, xu, encoding, rng, problem, self.cfg.get("initializer"))
        F = np.asarray(backend.evaluate(X, problem).F, dtype=float)

        variation = _build_variation(self.cfg, encoding, xl, xu, problem)
        archive = _build_archive(self.cfg, seed)
        if archive is not None:
            archive.add(X, F, X.shape[0])
        selection_ranks, selection_crowding = self.kernel.nsga2_ranking(F)

        result_mode = str(self.cfg.get("result_mode", "non_dominated")).strip().lower()
        if result_mode not in {"non_dominated", "population"}:
            raise ValueError("result_mode must be one of: non_dominated, population")

        self._st = AGEMOEAState(
            X=X,
            F=F,
            G=None,
            rng=rng,
            pop_size=pop_size,
            n_eval=X.shape[0],
            generation=0,
            max_evals=max_evals,
            variation=variation,
            archive=archive,
            selection_ranks=np.asarray(selection_ranks, dtype=int),
            selection_crowding=np.asarray(selection_crowding, dtype=float),
            result_mode=result_mode,
        )

    def ask(self) -> np.ndarray:
        """Generate offspring for external evaluation.

        Returns
        -------
        np.ndarray
            Offspring decision variables, shape ``(n_offspring, n_var)``.

        Raises
        ------
        RuntimeError
            If called before ``initialize()`` or previous offspring not consumed.
        """
        if self._st is None:
            raise RuntimeError("Algorithm not initialized. Call initialize() first.")
        if self._st.pending_offspring is not None:
            raise RuntimeError("Previous offspring not yet consumed by tell().")

        st = self._st
        if st.selection_ranks is None or st.selection_crowding is None or st.selection_ranks.shape[0] != st.F.shape[0]:
            self._refresh_selection_metrics(st)
        assert st.selection_ranks is not None
        assert st.selection_crowding is not None
        n_parents = 2 * (st.pop_size // 2)
        parents_idx = self.kernel.tournament_selection(
            ranks=st.selection_ranks,
            crowding=st.selection_crowding,
            pressure=2,
            rng=st.rng,
            n_parents=n_parents,
        )

        assert st.variation is not None
        X_off = st.variation.produce_offspring(st.X[parents_idx], st.rng)
        st.pending_offspring = X_off
        return np.array(X_off, copy=True)

    def tell(self, eval_result: Any, problem: ProblemProtocol | None = None) -> bool:
        """Receive evaluated offspring and update population.

        Parameters
        ----------
        eval_result : Any
            Objective values as ``np.ndarray``, or an object with ``.F`` attribute,
            or a dict with ``"F"`` key.
        problem : ProblemProtocol | None
            Unused, kept for interface consistency.

        Returns
        -------
        bool
            Always ``False`` (AGE-MOEA has no early-stop criterion).

        Raises
        ------
        RuntimeError
            If called before ``ask()``.
        """
        if self._st is None or self._st.pending_offspring is None:
            raise RuntimeError("No pending offspring. Call ask() first.")

        st = self._st
        X_off = st.pending_offspring
        assert X_off is not None

        if hasattr(eval_result, "F"):
            F_off = np.asarray(eval_result.F, dtype=float)
        elif isinstance(eval_result, dict):
            F_off = np.asarray(eval_result["F"], dtype=float)
        else:
            F_off = np.asarray(eval_result, dtype=float)

        st.n_eval += X_off.shape[0]

        if st.archive is not None:
            st.archive.add(X_off, F_off, st.n_eval)

        X_combined = np.vstack([st.X, X_off])
        F_combined = np.vstack([st.F, F_off])

        survivors = _age_survival(F_combined, st.pop_size, self.kernel)
        st.X = X_combined[survivors]
        st.F = F_combined[survivors]
        self._refresh_selection_metrics(st)

        st.pending_offspring = None
        st.generation += 1
        return False

    def should_terminate(self) -> bool:
        """Check if termination criterion is met."""
        if self._st is None:
            return True
        return self._st.n_eval >= self._st.max_evals

    def result(self) -> dict[str, Any]:
        """Get optimization result.

        Returns
        -------
        dict
            Result dictionary with ``X``, ``F``, ``n_eval``, ``n_gen``,
            ``population``, and optionally ``archive``.
        """
        if self._st is None:
            raise RuntimeError("Algorithm not initialized.")
        return build_agemoea_result(self._st, kernel=self.kernel)

    @property
    def state(self) -> AGEMOEAState | None:
        """Access current algorithm state."""
        return self._st
