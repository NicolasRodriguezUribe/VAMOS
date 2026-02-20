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
    from vamos.archive.bounded_archive import BoundedArchive

import numpy as np

from vamos.engine.algorithm.components.population import initialize_population, resolve_bounds
from vamos.engine.algorithm.components.variation.helpers import (
    ensure_supported_operator_names,
    ensure_supported_repair_name,
)
from vamos.engine.algorithm.components.variation.pipeline import VariationPipeline
from vamos.engine.config.variation import (
    ensure_operator_tuple,
    ensure_operator_tuple_optional,
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


def _build_variation(config: dict[str, Any], encoding: Any, xl: Any, xu: Any, problem: ProblemProtocol) -> VariationPipeline:
    explicit_overrides: dict[str, Any] = {}
    if "crossover" in config:
        explicit_overrides["crossover"] = config["crossover"]
    if "mutation" in config:
        explicit_overrides["mutation"] = config["mutation"]
    if "repair" in config:
        explicit_overrides["repair"] = config["repair"]

    var_cfg = resolve_default_variation_config(encoding, explicit_overrides)
    c_name, c_kwargs = ensure_operator_tuple(var_cfg.get("crossover", ("sbx", {})), key="crossover")
    m_name, m_kwargs = ensure_operator_tuple(var_cfg.get("mutation", ("pm", {})), key="mutation")
    repair_tuple = ensure_operator_tuple_optional(var_cfg.get("repair"), key="repair")
    cross_name, mut_name = ensure_supported_operator_names(encoding, c_name, m_name)
    repair_cfg = None
    if repair_tuple is not None:
        repair_name, repair_params = repair_tuple
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


def _build_archive(config: dict[str, Any], seed: int) -> BoundedArchive | None:
    from vamos.archive import ExternalArchiveConfig
    from vamos.archive.bounded_archive import BoundedArchive, BoundedArchiveConfig

    ext_cfg = config.get("external_archive")
    if ext_cfg is None:
        return None
    if isinstance(ext_cfg, dict):
        ext_cfg = ExternalArchiveConfig(**ext_cfg)
    if ext_cfg.capacity is None or ext_cfg.capacity <= 0:
        return None
    bac = BoundedArchiveConfig(
        size_cap=ext_cfg.capacity,
        truncate_size=ext_cfg.truncate_size,
        archive_type=ext_cfg.archive_type,
        prune_policy=ext_cfg.pruning,
        epsilon=ext_cfg.epsilon,
        rng_seed=seed,
        nondominated_only=ext_cfg.nondominated_only,
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
        ranks, crowding = self.kernel.nsga2_ranking(st.F)
        n_parents = 2 * (st.pop_size // 2)
        parents_idx = self.kernel.tournament_selection(
            ranks=ranks,
            crowding=crowding,
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
