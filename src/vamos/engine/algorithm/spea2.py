"""SPEA2 (Strength Pareto Evolutionary Algorithm 2).

Enhanced implementation with:
- Ask/Tell interface for external evaluation control
- HV-based early termination
- Live visualization callbacks
- External archive support (crowding/hypervolume)
- Full constraint handling

Key Features:
    - Strength-based fitness incorporating dominance information
    - k-th nearest neighbor density estimation
    - Archive truncation preserving boundary solutions
    - Support for continuous, binary, and integer encodings

References:
    E. Zitzler, M. Laumanns, and L. Thiele, "SPEA2: Improving the Strength
    Pareto Evolutionary Algorithm," TIK-Report 103, ETH Zurich, 2001.

Example:
    Basic usage with run():
    >>> from vamos import optimize, SPEA2Config, ZDT1
    >>> result = optimize(
    ...     problem=ZDT1(n_var=30),
    ...     algorithm="spea2",
    ...     algorithm_config=SPEA2Config().pop_size(100).archive_size(100).fixed(),
    ...     termination=("n_eval", 10000),
    ... )

    Ask/Tell interface:
    >>> from vamos.engine.algorithm.spea2 import SPEA2
    >>> from vamos.foundation.kernel.registry import resolve_kernel
    >>> kernel = resolve_kernel("numpy")
    >>> spea2 = SPEA2({"pop_size": 100}, kernel)
    >>> spea2.initialize(problem, ("n_eval", 10000), seed=42)
    >>> while not spea2.should_terminate():
    ...     X_off = spea2.ask()
    ...     F_off = problem.evaluate(X_off)
    ...     spea2.tell(EvalResult(F=F_off, G=None))
    >>> result = spea2.result()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.engine.algorithm.components.base import (
    get_eval_backend,
    get_live_viz,
    notify_generation,
    parse_termination,
    resolve_archive_size,
    setup_archive,
    setup_hv_tracker,
    update_archive,
)
from vamos.engine.algorithm.components.population import (
    evaluate_population,
    evaluate_population_with_constraints,
    initialize_population,
    resolve_bounds,
)
from vamos.engine.algorithm.spea2_state import SPEA2State, build_spea2_result
from vamos.engine.operators.real import PolynomialMutation, SBXCrossover, VariationWorkspace
from vamos.foundation.constraints.utils import compute_violation, is_feasible

if TYPE_CHECKING:
    from vamos.foundation.eval.backends import EvaluationBackend
    from vamos.foundation.kernel.backend import KernelBackend
    from vamos.foundation.problem.types import ProblemProtocol
    from vamos.ux.visualization.live_viz import LiveVisualization

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SPEA2 helper functions
# ---------------------------------------------------------------------------


def _dominance_matrix(
    F: np.ndarray, G: np.ndarray | None, constraint_mode: str
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Compute dominance matrix with optional feasibility-aware handling."""
    n = F.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=bool), None, None

    feas = None
    cv = None

    if constraint_mode and constraint_mode != "none" and G is not None:
        cv = compute_violation(G)
        feas = is_feasible(G)

    dom = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Feasibility-based dominance
            if feas is not None:
                if feas[i] and not feas[j]:
                    dom[i, j] = True
                    continue
                if not feas[i] and feas[j]:
                    continue
                if not feas[i] and not feas[j]:
                    if cv[i] < cv[j]:
                        dom[i, j] = True
                    continue
            # Standard Pareto dominance
            if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                dom[i, j] = True

    return dom, feas, cv


def _spea2_fitness(
    F: np.ndarray, dom: np.ndarray, k: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Compute SPEA2 fitness and distance matrix."""
    n = F.shape[0]
    if n == 0:
        return np.empty(0), np.empty((0, 0))

    if k is None:
        k = max(1, int(np.sqrt(n)))
    k = min(k, n - 1) if n > 1 else 1

    # Strength: number of solutions each solution dominates
    strength = dom.sum(axis=1)

    # Raw fitness: sum of strengths of all dominators
    raw_fitness = np.zeros(n)
    for i in range(n):
        dominators = np.where(dom[:, i])[0]
        raw_fitness[i] = strength[dominators].sum()

    # Distance matrix in objective space
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(F[i] - F[j])
            dist[i, j] = d
            dist[j, i] = d

    # Density: based on k-th nearest neighbor distance
    if n == 1:
        density = np.array([0.0])
    else:
        density = np.zeros(n)
        for i in range(n):
            sorted_dists = np.sort(dist[i])
            sigma_k = sorted_dists[k] if k < n else sorted_dists[-1]
            density[i] = 1.0 / (sigma_k + 2.0)

    return raw_fitness + density, dist


def _truncate_by_distance(dist_matrix: np.ndarray, keep: int) -> np.ndarray:
    """Truncate by iteratively removing solution with smallest distance."""
    candidates = list(range(dist_matrix.shape[0]))
    if len(candidates) <= keep:
        return np.asarray(candidates, dtype=int)

    dist = dist_matrix.copy()
    while len(candidates) > keep:
        sub = dist[np.ix_(candidates, candidates)]
        np.fill_diagonal(sub, np.inf)
        nearest = np.partition(sub, 1, axis=1)[:, 1]
        remove_pos = int(np.argmin(nearest))
        del candidates[remove_pos]

    return np.asarray(candidates, dtype=int)


def _environmental_selection(
    X: np.ndarray,
    F: np.ndarray,
    G: np.ndarray | None,
    archive_size: int,
    k_neighbors: int | None,
    constraint_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """SPEA2 environmental selection."""
    dom, _, _ = _dominance_matrix(F, G, constraint_mode)
    fitness, dist = _spea2_fitness(F, dom, k_neighbors)

    # Select non-dominated individuals (fitness < 1)
    selected = np.flatnonzero(fitness < 1.0)
    if selected.size == 0:
        order = np.argsort(fitness)
        selected = order[: min(archive_size, F.shape[0])]

    # Truncate if too many non-dominated
    if selected.size > archive_size:
        rel = _truncate_by_distance(dist[np.ix_(selected, selected)], archive_size)
        selected = selected[rel]
    # Fill if too few non-dominated
    elif selected.size < archive_size:
        remaining = np.setdiff1d(np.arange(F.shape[0]), selected, assume_unique=True)
        if remaining.size:
            order = remaining[np.argsort(fitness[remaining])]
            needed = archive_size - selected.size
            selected = np.concatenate([selected, order[:needed]])

    return X[selected], F[selected], G[selected] if G is not None else None


# ---------------------------------------------------------------------------
# SPEA2 Algorithm
# ---------------------------------------------------------------------------


class SPEA2:
    """SPEA2 (Strength Pareto Evolutionary Algorithm 2).

    Enhanced implementation with ask/tell interface, HV termination,
    and live visualization support.

    Parameters
    ----------
    config : dict
        Algorithm configuration with keys:
        - pop_size (int): Population size
        - archive_size (int, optional): Internal archive size (default: pop_size)
        - crossover (dict): Crossover operator config
        - mutation (dict): Mutation operator config
        - k_neighbors (int, optional): k for density estimation (default: sqrt(N))
        - constraint_mode (str, optional): "none" or "feasibility"
    kernel : KernelBackend
        Backend for vectorized operations.
    """

    def __init__(self, config: dict[str, Any], kernel: "KernelBackend") -> None:
        self.cfg = config
        self.kernel = kernel
        self._st: SPEA2State | None = None

    def run(
        self,
        problem: "ProblemProtocol",
        termination: tuple[str, Any],
        seed: int,
        eval_backend: "EvaluationBackend | None" = None,
        live_viz: "LiveVisualization | None" = None,
    ) -> dict[str, Any]:
        """Run SPEA2 optimization."""
        live_cb, eval_backend, max_eval, hv_tracker = self._initialize_run(
            problem, termination, seed, eval_backend, live_viz
        )
        st = self._st
        assert st is not None, "State not initialized"

        generation = 0
        live_cb.on_generation(generation, F=st.env_F)
        hv_reached = hv_tracker.enabled and hv_tracker.reached(st.env_F)

        while st.n_eval < max_eval and not hv_reached:
            st.generation = generation
            X_off = self.ask()

            # Evaluate offspring
            eval_result = eval_backend.evaluate(X_off, problem)
            hv_reached = self.tell(eval_result, problem)

            if hv_tracker.enabled and hv_tracker.reached(st.env_F):
                hv_reached = True
                break

            generation += 1
            st.generation = generation
            notify_generation(live_cb, self.kernel, generation, st.env_F)

        result = build_spea2_result(st, hv_reached)
        live_cb.on_end(final_F=st.env_F)
        return result

    def _initialize_run(
        self,
        problem: "ProblemProtocol",
        termination: tuple[str, Any],
        seed: int,
        eval_backend: "EvaluationBackend | None" = None,
        live_viz: "LiveVisualization | None" = None,
    ) -> tuple[Any, Any, int, Any]:
        """Initialize algorithm state for a run."""
        max_eval, hv_config = parse_termination(termination, "SPEA2")

        eval_backend = get_eval_backend(eval_backend)
        live_cb = get_live_viz(live_viz)
        rng = np.random.default_rng(seed)

        pop_size = int(self.cfg.get("pop_size", 100))
        env_archive_size = int(self.cfg.get("archive_size", pop_size))
        offspring_size = pop_size
        k_neighbors = self.cfg.get("k_neighbors")
        constraint_mode = self.cfg.get("constraint_mode", "none")

        encoding = getattr(problem, "encoding", "continuous")
        n_var = problem.n_var
        n_obj = problem.n_obj
        xl, xu = resolve_bounds(problem, encoding)

        # Initialize population
        initializer_cfg = self.cfg.get("initializer")
        X = initialize_population(
            pop_size, n_var, xl, xu, encoding, rng, problem, initializer=initializer_cfg
        )

        if constraint_mode and constraint_mode != "none":
            F, G = evaluate_population_with_constraints(problem, X)
        else:
            F = evaluate_population(problem, X)
            G = None
        n_eval = pop_size

        # Environmental selection for initial internal archive
        env_X, env_F, env_G = _environmental_selection(
            X, F, G, env_archive_size, k_neighbors, constraint_mode
        )

        # Setup external archive (optional, separate from internal)
        ext_archive_size = resolve_archive_size(self.cfg)
        archive_type = self.cfg.get("archive_type", "crowding")
        archive_X, archive_F, archive_manager = setup_archive(
            self.kernel, env_X, env_F, n_var, n_obj, X.dtype, ext_archive_size, archive_type
        )

        # Setup HV tracker
        hv_tracker = setup_hv_tracker(hv_config, self.kernel)

        # Build variation operators
        crossover_fn, mutation_fn = self._build_variation(
            encoding, n_var, xl, xu, rng
        )

        live_cb.on_start(problem=problem, algorithm=self, config=self.cfg)

        # Create state
        self._st = SPEA2State(
            X=X,
            F=F,
            G=G,
            rng=rng,
            pop_size=pop_size,
            offspring_size=offspring_size,
            constraint_mode=constraint_mode,
            generation=0,
            n_eval=n_eval,
            # External archive (from base class)
            archive_size=ext_archive_size,
            archive_X=archive_X,
            archive_F=archive_F,
            archive_manager=archive_manager,
            # HV tracking
            hv_tracker=hv_tracker,
            # SPEA2-specific internal archive
            env_X=env_X,
            env_F=env_F,
            env_G=env_G,
            env_archive_size=env_archive_size,
            k_neighbors=k_neighbors,
            crossover_fn=crossover_fn,
            mutation_fn=mutation_fn,
            xl=xl,
            xu=xu,
        )

        return live_cb, eval_backend, max_eval, hv_tracker

    def _build_variation(
        self,
        encoding: str,
        n_var: int,
        xl: np.ndarray,
        xu: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[Any, Any]:
        """Build variation operators based on encoding.

        Returns
        -------
        tuple
            (crossover_fn, mutation_fn) callables
        """
        # Unpack crossover config (format: ("sbx", {"prob": 0.9, "eta": 20.0}))
        cross_cfg = self.cfg.get("crossover", ("sbx", {}))
        if isinstance(cross_cfg, tuple):
            cross_method, cross_params = cross_cfg
            cross_params = dict(cross_params) if cross_params else {}
        else:
            cross_params = cross_cfg or {}

        # Unpack mutation config (format: ("pm", {"prob": "1/n", "eta": 20.0}))
        mut_cfg = self.cfg.get("mutation", ("pm", {}))
        if isinstance(mut_cfg, tuple):
            mut_method, mut_params = mut_cfg
            mut_params = dict(mut_params) if mut_params else {}
        else:
            mut_params = mut_cfg or {}

        # Prepare mutation probability
        mut_prob = mut_params.get("prob", 1.0 / n_var)
        if isinstance(mut_prob, str):
            mut_prob = 1.0 / n_var if "1/n" in mut_prob else float(mut_prob)

        workspace = VariationWorkspace()

        # For now, use SBX+PM for all encodings
        crossover_operator = SBXCrossover(
            prob_crossover=cross_params.get("prob", 0.9),
            eta=cross_params.get("eta", 20.0),
            lower=xl,
            upper=xu,
            workspace=workspace,
            allow_inplace=True,
        )
        mutation_operator = PolynomialMutation(
            prob_mutation=mut_prob,
            eta=mut_params.get("eta", 20.0),
            lower=xl,
            upper=xu,
            workspace=workspace,
        )

        crossover_fn = lambda parents, rng=rng: crossover_operator(parents, rng)
        mutation_fn = lambda X_child, rng=rng: mutation_operator(X_child, rng)

        return crossover_fn, mutation_fn

    # -------------------------------------------------------------------------
    # Ask/Tell Interface
    # -------------------------------------------------------------------------

    def initialize(
        self,
        problem: "ProblemProtocol",
        termination: tuple[str, Any],
        seed: int,
        eval_backend: "EvaluationBackend | None" = None,
        live_viz: "LiveVisualization | None" = None,
    ) -> None:
        """Initialize algorithm for ask/tell loop."""
        self._live_cb, self._eval_backend, self._max_eval, self._hv_tracker = (
            self._initialize_run(problem, termination, seed, eval_backend, live_viz)
        )
        self._problem = problem
        if self._st is not None:
            self._live_cb.on_generation(0, F=self._st.env_F)

    def ask(self) -> np.ndarray:
        """Generate offspring for external evaluation."""
        if self._st is None:
            raise RuntimeError("Call initialize() or run() before ask()")

        st = self._st
        n_pairs = st.offspring_size

        # Random selection from internal archive
        # SPEA2 uses fitness-based tournament but for simplicity we use random
        archive_size = st.env_X.shape[0]
        parent_idx = st.rng.integers(0, archive_size, size=(n_pairs, 2))

        # Gather parents in shape (n_pairs, 2, n_var)
        n_var = st.env_X.shape[1]
        parents = st.env_X[parent_idx.reshape(-1)].reshape(n_pairs, 2, n_var)

        # Apply crossover
        offspring = st.crossover_fn(parents, st.rng)

        # Take first child from each pair
        offspring_X = offspring[:, 0, :].copy()

        # Apply mutation
        offspring_X = st.mutation_fn(offspring_X, st.rng)

        # Clip to bounds
        np.clip(offspring_X, st.xl, st.xu, out=offspring_X)

        st.pending_offspring = offspring_X
        return offspring_X

    def tell(
        self,
        eval_result: Any,
        problem: "ProblemProtocol | None" = None,
    ) -> bool:
        """Process evaluated offspring."""
        if self._st is None or self._st.pending_offspring is None:
            raise RuntimeError("Call ask() before tell()")

        st = self._st
        offspring_X = st.pending_offspring

        # Extract F and G from result
        if hasattr(eval_result, "F"):
            F = eval_result.F
            G = getattr(eval_result, "G", None)
        elif isinstance(eval_result, dict):
            F = eval_result.get("F")
            G = eval_result.get("G")
        else:
            F = eval_result
            G = None

        # Update evaluation count
        st.n_eval += len(F)

        # Combine archive and offspring for environmental selection
        X_union = np.vstack([st.env_X, offspring_X])
        F_union = np.vstack([st.env_F, F])

        if G is not None:
            if st.env_G is not None:
                G_union = np.vstack([st.env_G, G])
            else:
                G_union = G
        elif st.env_G is not None:
            G_union = np.vstack(
                [st.env_G, np.zeros((len(F), st.env_G.shape[1]))]
            )
        else:
            G_union = None

        # Environmental selection
        st.env_X, st.env_F, st.env_G = _environmental_selection(
            X_union,
            F_union,
            G_union,
            st.env_archive_size,
            st.k_neighbors,
            st.constraint_mode,
        )

        # Update population reference
        st.X = st.env_X
        st.F = st.env_F
        st.G = st.env_G

        # Update external archive
        update_archive(st, st.env_X, st.env_F)

        # Clear pending
        st.pending_offspring = None

        # Check HV termination
        if st.hv_tracker is not None and st.hv_tracker.enabled:
            return st.hv_tracker.reached(st.env_F)

        return False

    def should_terminate(self) -> bool:
        """Check if termination criterion is met."""
        if self._st is None:
            return True

        if hasattr(self, "_max_eval"):
            if self._st.n_eval >= self._max_eval:
                return True

        if hasattr(self, "_hv_tracker") and self._hv_tracker.enabled:
            return self._hv_tracker.reached(self._st.env_F)

        return False

    def result(self) -> dict[str, Any]:
        """Get final optimization result."""
        if self._st is None:
            raise RuntimeError("Call initialize() and run before result()")

        hv_reached = (
            hasattr(self, "_hv_tracker")
            and self._hv_tracker.enabled
            and self._hv_tracker.reached(self._st.env_F)
        )
        return build_spea2_result(self._st, hv_reached)

    def _selection_metrics(
        self, F: np.ndarray, G: np.ndarray | None, constraint_mode: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute selection metrics for mating selection."""
        if F.size == 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=float)

        if constraint_mode and constraint_mode != "none" and G is not None:
            cv = compute_violation(G)
            feas = is_feasible(G)
            if feas.any():
                feas_idx = np.nonzero(feas)[0]
                ranks, crowd = self.kernel.nsga2_ranking(F[feas_idx])
                metrics_rank = np.full(F.shape[0], ranks.max(initial=0) + 1, dtype=int)
                metrics_crowd = np.zeros(F.shape[0], dtype=float)
                metrics_rank[feas_idx] = ranks
                metrics_crowd[feas_idx] = crowd
                metrics_crowd[~feas] = -cv[~feas]
                return metrics_rank, metrics_crowd
            ranks = np.zeros(F.shape[0], dtype=int)
            crowd = -cv
            return ranks, crowd

        return self.kernel.nsga2_ranking(F)
