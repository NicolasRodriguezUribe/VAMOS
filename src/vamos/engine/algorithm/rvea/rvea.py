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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vamos.archive.bounded_archive import BoundedArchive

import numpy as np

from vamos.engine.algorithm.components.population import initialize_population, resolve_bounds
from vamos.engine.algorithm.components.variation.pipeline import VariationPipeline
from vamos.engine.algorithm.components.variation.helpers import (
    ensure_supported_operator_names,
    ensure_supported_repair_name,
)
from vamos.engine.config.variation import (
    ensure_operator_tuple,
    ensure_operator_tuple_optional,
    resolve_default_variation_config,
)
from vamos.foundation.encoding import normalize_encoding
from vamos.foundation.eval.backends import EvaluationBackend, SerialEvalBackend
from vamos.foundation.kernel.backend import KernelBackend
from vamos.foundation.kernel import default_kernel
from vamos.foundation.problem.types import ProblemProtocol

from .state import RVEAState, build_rvea_result


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def _generate_reference_vectors(n_obj: int, n_partitions: int = 12) -> np.ndarray:
    """Generate uniformly distributed reference vectors using Das-Dennis."""
    from itertools import combinations

    def _das_dennis(n_partitions: int, n_obj: int) -> np.ndarray:
        if n_obj == 1:
            return np.array([[1.0]])
        compositions: list[list[int]] = []
        for indices in combinations(range(n_partitions + n_obj - 1), n_obj - 1):
            prev = -1
            composition: list[int] = []
            for idx in indices:
                composition.append(idx - prev - 1)
                prev = idx
            composition.append(n_partitions + n_obj - 2 - prev)
            compositions.append(composition)
        vectors = np.asarray(compositions, dtype=float) / float(n_partitions)
        return vectors

    ref_dirs = _das_dennis(n_partitions, n_obj)
    return ref_dirs


def _calc_V(ref_dirs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(ref_dirs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return np.asarray(ref_dirs / norms, dtype=float)


def _calc_gamma(V: np.ndarray) -> np.ndarray:
    cosine = V @ V.T
    gamma = np.arccos((-np.sort(-1.0 * cosine, axis=1))[:, 1])
    gamma = np.maximum(gamma, 1e-64)
    return np.asarray(gamma, dtype=float)


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

    survivor_indices: list[int] = []
    for k in range(len(V)):
        assigned = np.where(niches == k)[0]
        if assigned.size == 0:
            continue
        theta = acute_angle[assigned, k]
        M = float(n_obj) if n_obj > 2 else 1.0
        penalty = M * ((n_gen / n_max_gen) ** alpha) * (theta / gamma[k])
        apd = dist_to_ideal[assigned] * (1.0 + penalty)
        survivor = assigned[int(np.argmin(apd))]
        survivor_indices.append(int(survivor))

    survivors = np.asarray(survivor_indices, dtype=int)
    nadir = F[survivors].max(axis=0) if survivors.size else None
    return survivors, ideal, nadir


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
        archive_type=ext_cfg.archive_type,
        prune_policy=ext_cfg.pruning,
        epsilon=ext_cfg.epsilon,
        rng_seed=seed,
        nondominated_only=ext_cfg.nondominated_only,
    )
    return BoundedArchive(bac)


class RVEA:
    """RVEA: Reference Vector-guided Evolutionary Algorithm.

    Implements angle-penalized distance (APD) survival and periodic
    reference-vector adaptation.

    Parameters
    ----------
    config : dict
        Algorithm configuration.
    kernel : KernelBackend, optional
        Backend for vectorized operations.

    Examples
    --------
    Batch mode:

    >>> algo = RVEA(config, kernel)
    >>> result = algo.run(problem, ("max_evaluations", 10000), seed=42)

    Ask/tell interface:

    >>> algo = RVEA(config, kernel)
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
        self._st: RVEAState | None = None

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
        """Run RVEA optimization."""
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
        n_obj = int(problem.n_obj)
        n_partitions = int(self.cfg.get("n_partitions", 12))
        alpha = float(self.cfg.get("alpha", 2.0))
        adapt_freq = self.cfg.get("adapt_freq", 0.1)

        term_key, term_val = termination
        if term_key == "n_gen":
            max_gen = max(1, int(term_val))
            max_evals = max_gen * pop_size
        elif term_key == "max_evaluations":
            max_evals = int(term_val)
            max_gen = max(1, int(math.ceil((max_evals - pop_size) / pop_size)))
        else:
            raise ValueError("Unsupported termination criterion for RVEA.")

        ref_dirs = _generate_reference_vectors(n_obj, n_partitions)
        if ref_dirs.shape[0] != pop_size:
            raise ValueError(
                f"RVEA requires pop_size == #ref_dirs for partitions={n_partitions} (pop_size={pop_size}, ref_dirs={ref_dirs.shape[0]})."
            )
        V = _calc_V(ref_dirs)
        gamma = _calc_gamma(V)

        encoding = normalize_encoding(getattr(problem, "encoding", "real"))
        xl, xu = resolve_bounds(problem, encoding)
        X = initialize_population(pop_size, problem.n_var, xl, xu, encoding, rng, problem, self.cfg.get("initializer"))
        F = np.asarray(backend.evaluate(X, problem).F, dtype=float)

        variation = _build_variation(self.cfg, encoding, xl, xu, problem)
        archive = _build_archive(self.cfg, seed)
        if archive is not None:
            archive.add(X, F, X.shape[0])

        adapt_interval = None
        if adapt_freq is not None:
            adapt_interval = max(1, int(math.ceil(max_gen * float(adapt_freq))))

        result_mode = str(self.cfg.get("result_mode", "non_dominated")).strip().lower()
        if result_mode not in {"non_dominated", "population"}:
            raise ValueError("result_mode must be one of: non_dominated, population")

        self._st = RVEAState(
            X=X,
            F=F,
            G=None,
            rng=rng,
            pop_size=pop_size,
            n_eval=X.shape[0],
            generation=1,
            max_evals=max_evals,
            max_gen=max_gen,
            variation=variation,
            archive=archive,
            ref_dirs=ref_dirs,
            V=V,
            gamma=gamma,
            ideal=np.full(n_obj, np.inf),
            nadir=None,
            alpha=alpha,
            adapt_interval=adapt_interval,
            n_obj=n_obj,
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
        assert st.variation is not None
        parents_idx = st.rng.integers(0, len(st.X), size=st.pop_size)
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
            Always ``False`` (RVEA has no early-stop criterion).

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

        survivors, st.ideal, st.nadir = _apd_survival(
            F_combined,
            st.V,
            st.gamma,
            st.ideal,
            st.pop_size,
            st.generation,
            st.max_gen,
            st.alpha,
        )
        if survivors.size == 0:
            st.pending_offspring = None
            st.generation += 1
            return False

        st.X = X_combined[survivors]
        st.F = F_combined[survivors]

        # Periodic reference-vector adaptation
        if st.adapt_interval is not None and st.generation % st.adapt_interval == 0 and st.nadir is not None:
            scale = np.maximum(st.nadir - st.ideal, 1e-64)
            st.V = _calc_V(_calc_V(st.ref_dirs) * scale)
            st.gamma = _calc_gamma(st.V)

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
        return build_rvea_result(self._st, kernel=self.kernel)

    @property
    def state(self) -> RVEAState | None:
        """Access current algorithm state."""
        return self._st
