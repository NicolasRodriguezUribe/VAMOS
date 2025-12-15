"""MOEA/D: Multi-Objective Evolutionary Algorithm based on Decomposition.

This module implements MOEA/D, which decomposes a multi-objective optimization
problem into a number of scalar subproblems using weight vectors and optimizes
them simultaneously. Each subproblem is optimized using information from its
neighboring subproblems.

Key Features:
    - Tchebyshev, weighted sum, PBI, and modified Tchebyshev scalarization
    - Adaptive neighborhood-based mating restriction
    - Support for continuous, binary, and integer encodings
    - Feasibility-aware constraint handling
    - Ask/tell interface for external evaluation
    - HV-based termination and live visualization support
    - External archive support

References:
    Q. Zhang and H. Li, "MOEA/D: A Multiobjective Evolutionary Algorithm Based on
    Decomposition," IEEE Trans. Evolutionary Computation, vol. 11, no. 6, 2007.

Example:
    >>> from vamos import optimize, MOEADConfig, ZDT1
    >>> result = optimize(
    ...     problem=ZDT1(n_var=30),
    ...     algorithm="moead",
    ...     algorithm_config=MOEADConfig().pop_size(100).fixed(),
    ...     termination=("n_eval", 10000),
    ... )
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
    evaluate_population_with_constraints,
    resolve_bounds,
)
from vamos.engine.algorithm.components.utils import resolve_bounds_array, resolve_prob_expression
from vamos.engine.algorithm.components.weight_vectors import load_or_generate_weight_vectors
from vamos.engine.algorithm.moead_state import MOEADState, build_moead_result
from vamos.engine.operators.binary import (
    bit_flip_mutation,
    one_point_crossover,
    random_binary_population,
    two_point_crossover,
    uniform_crossover,
)
from vamos.engine.operators.integer import (
    arithmetic_integer_crossover,
    creep_mutation,
    random_integer_population,
    random_reset_mutation,
    uniform_integer_crossover,
)
from vamos.engine.operators.real import PolynomialMutation, SBXCrossover, VariationWorkspace
from vamos.foundation.constraints.utils import compute_violation, is_feasible

if TYPE_CHECKING:
    from vamos.foundation.eval.backends import EvaluationBackend
    from vamos.foundation.kernel.backend import KernelBackend
    from vamos.foundation.problem.types import ProblemProtocol
    from vamos.ux.visualization.live_viz import LiveVisualization

_logger = logging.getLogger(__name__)


# Operator registries
_BINARY_CROSSOVER = {
    "one_point": one_point_crossover,
    "single_point": one_point_crossover,
    "1point": one_point_crossover,
    "two_point": two_point_crossover,
    "2point": two_point_crossover,
    "uniform": uniform_crossover,
}

_BINARY_MUTATION = {
    "bitflip": bit_flip_mutation,
    "bit_flip": bit_flip_mutation,
}

_INT_CROSSOVER = {
    "uniform": uniform_integer_crossover,
    "blend": arithmetic_integer_crossover,
    "arithmetic": arithmetic_integer_crossover,
}

_INT_MUTATION = {
    "reset": random_reset_mutation,
    "random_reset": random_reset_mutation,
    "creep": creep_mutation,
}


class MOEAD:
    """
    Multi-Objective Evolutionary Algorithm based on Decomposition.

    MOEA/D decomposes a multi-objective problem into scalar subproblems using
    weight vectors and optimizes them collaboratively via neighborhood-based
    mating and replacement.

    Parameters
    ----------
    config : dict
        Algorithm configuration with keys:
        - pop_size (int): Population size (should match number of weight vectors)
        - crossover (tuple): Crossover operator config, e.g., ("sbx", {"prob": 0.9})
        - mutation (tuple): Mutation operator config, e.g., ("pm", {"prob": "1/n"})
        - weight_vectors (dict, optional): {"path": str, "divisions": int}
        - neighbor_size (int, optional): T parameter (default: 20)
        - delta (float, optional): Neighborhood selection probability (default: 0.9)
        - replace_limit (int, optional): Max replacements per offspring (default: 2)
        - aggregation (tuple, optional): ("tchebycheff", {}) or similar
        - constraint_mode (str, optional): "feasibility" or "none"
        - archive (dict, optional): {"size": int, "type": str}
    kernel : KernelBackend
        Backend for vectorized operations.

    Attributes
    ----------
    cfg : dict
        Stored configuration.
    kernel : KernelBackend
        Kernel backend instance.

    Examples
    --------
    >>> from vamos import MOEADConfig
    >>> config = MOEADConfig().pop_size(100).crossover("sbx", prob=0.9).fixed()
    >>> moead = MOEAD(config, kernel)
    >>> result = moead.run(problem, ("n_eval", 10000), seed=42)

    Using ask/tell for external evaluation:
    >>> moead._initialize_run(problem, ("n_eval", 10000), seed=42)
    >>> while moead._st.n_eval < 10000:
    ...     X_off = moead.ask()
    ...     F_off = my_external_evaluator(X_off)
    ...     moead.tell(EvalResult(F=F_off, G=None))
    """

    def __init__(self, config: dict[str, Any], kernel: "KernelBackend") -> None:
        self.cfg = config
        self.kernel = kernel
        self._st: MOEADState | None = None

    def run(
        self,
        problem: "ProblemProtocol",
        termination: tuple[str, Any],
        seed: int,
        eval_backend: "EvaluationBackend | None" = None,
        live_viz: "LiveVisualization | None" = None,
    ) -> dict[str, Any]:
        """
        Run the MOEA/D algorithm.

        Parameters
        ----------
        problem : ProblemProtocol
            The optimization problem to solve.
        termination : tuple[str, Any]
            Termination criterion: ("n_eval", N) or ("hv", {...}).
        seed : int
            Random seed for reproducibility.
        eval_backend : EvaluationBackend | None
            Optional evaluation backend for parallel evaluation.
        live_viz : LiveVisualization | None
            Optional live visualization callback.

        Returns
        -------
        dict[str, Any]
            Result dictionary with X, F, weights, evaluations, and optional archive.
        """
        live_cb, eval_backend, max_eval, hv_tracker = self._initialize_run(
            problem, termination, seed, eval_backend, live_viz
        )
        st = self._st
        assert st is not None, "State not initialized"

        generation = 0
        live_cb.on_generation(generation, F=st.F)
        hv_reached = hv_tracker.enabled and hv_tracker.reached(st.hv_points())

        while st.n_eval < max_eval and not hv_reached:
            st.generation = generation
            X_off = self.ask()
            eval_result = eval_backend.evaluate(X_off, problem)
            hv_reached = self.tell(eval_result, problem)

            if hv_tracker.enabled and hv_tracker.reached(st.hv_points()):
                hv_reached = True
                break

            generation += 1
            st.generation = generation
            notify_generation(live_cb, self.kernel, generation, st.F)

        result = build_moead_result(st, hv_reached)
        live_cb.on_end(final_F=st.F)
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
        max_eval, hv_config = parse_termination(termination, "MOEA/D")

        eval_backend = get_eval_backend(eval_backend)
        live_cb = get_live_viz(live_viz)
        rng = np.random.default_rng(seed)

        pop_size = int(self.cfg["pop_size"])
        if pop_size < 2:
            raise ValueError("MOEA/D requires pop_size >= 2.")

        constraint_mode = self.cfg.get("constraint_mode", "feasibility")
        encoding = getattr(problem, "encoding", "continuous")
        xl, xu = resolve_bounds_array(problem, encoding)
        n_var = problem.n_var
        n_obj = problem.n_obj

        # Initialize population
        X, F, G = self._initialize_population(
            encoding, pop_size, n_var, xl, xu, rng, problem, constraint_mode
        )
        n_eval = pop_size

        # Setup weight vectors and neighborhoods
        weight_cfg = self.cfg.get("weight_vectors", {}) or {}
        weights = load_or_generate_weight_vectors(
            pop_size, n_obj,
            path=weight_cfg.get("path"),
            divisions=weight_cfg.get("divisions"),
        )

        neighbor_size = self.cfg.get("neighbor_size", min(20, pop_size))
        neighbor_size = max(2, min(neighbor_size, pop_size))
        neighbors = self._compute_neighbors(weights, neighbor_size)

        # Setup aggregation
        aggregation = self.cfg.get("aggregation", ("tchebycheff", {}))
        agg_method, agg_params = aggregation
        aggregator = self._build_aggregator(agg_method, agg_params)

        # Build variation operators
        crossover_fn, mutation_fn = self._build_variation_operators(
            encoding, n_var, xl, xu, rng
        )

        # Setup archive
        archive_size = resolve_archive_size(self.cfg)
        archive_type = self.cfg.get("archive_type", "crowding")
        archive_X, archive_F, archive_manager = setup_archive(
            self.kernel, X, F, n_var, n_obj, X.dtype, archive_size, archive_type
        )

        # Setup HV tracker
        hv_tracker = setup_hv_tracker(hv_config, self.kernel)

        live_cb.on_start(problem=problem, algorithm=self, config=self.cfg)

        # Create state
        self._st = MOEADState(
            X=X, F=F, G=G, rng=rng,
            pop_size=pop_size,
            offspring_size=pop_size,
            constraint_mode=constraint_mode,
            n_eval=n_eval,
            # MOEA/D-specific
            weights=weights,
            neighbors=neighbors,
            ideal=F.min(axis=0),
            aggregator=aggregator,
            neighbor_size=neighbor_size,
            delta=float(self.cfg.get("delta", 0.9)),
            replace_limit=max(1, int(self.cfg.get("replace_limit", 2))),
            crossover_fn=crossover_fn,
            mutation_fn=mutation_fn,
            xl=xl,
            xu=xu,
            # Archive
            archive_size=archive_size,
            archive_X=archive_X,
            archive_F=archive_F,
            archive_manager=archive_manager,
            # Termination
            hv_tracker=hv_tracker,
        )

        return live_cb, eval_backend, max_eval, hv_tracker

    def _initialize_population(
        self,
        encoding: str,
        pop_size: int,
        n_var: int,
        xl: np.ndarray,
        xu: np.ndarray,
        rng: np.random.Generator,
        problem: "ProblemProtocol",
        constraint_mode: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Initialize population based on encoding."""
        if encoding == "binary":
            X = random_binary_population(pop_size, n_var, rng)
        elif encoding == "integer":
            X = random_integer_population(pop_size, n_var, xl.astype(int), xu.astype(int), rng)
        else:
            X = rng.uniform(xl, xu, size=(pop_size, n_var))

        F, G = evaluate_population_with_constraints(problem, X)
        if constraint_mode == "none":
            G = None
        return X, F, G

    def _build_variation_operators(
        self,
        encoding: str,
        n_var: int,
        xl: np.ndarray,
        xu: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[Any, Any]:
        """Build crossover and mutation operators for the encoding."""
        cross_method, cross_params = self.cfg["crossover"]
        cross_params = dict(cross_params)

        mut_method, mut_params = self.cfg["mutation"]
        mut_params = dict(mut_params)
        if mut_params.get("prob") == "1/n":
            mut_params["prob"] = 1.0 / n_var

        if encoding == "binary":
            if cross_method not in _BINARY_CROSSOVER:
                raise ValueError(f"Unsupported MOEA/D crossover '{cross_method}' for binary encoding.")
            if mut_method not in _BINARY_MUTATION:
                raise ValueError(f"Unsupported MOEA/D mutation '{mut_method}' for binary encoding.")

            cross_fn = _BINARY_CROSSOVER[cross_method]
            cross_prob = float(cross_params.get("prob", 0.9))
            mut_fn = _BINARY_MUTATION[mut_method]
            mut_prob = resolve_prob_expression(mut_params.get("prob"), n_var, 1.0 / max(1, n_var))

            crossover = lambda parents, rng=rng: cross_fn(parents, cross_prob, rng)
            mutation = lambda X_child, rng=rng: mut_fn(X_child, mut_prob, rng) or X_child

        elif encoding == "integer":
            if cross_method not in _INT_CROSSOVER:
                raise ValueError(f"Unsupported MOEA/D crossover '{cross_method}' for integer encoding.")
            if mut_method not in _INT_MUTATION:
                raise ValueError(f"Unsupported MOEA/D mutation '{mut_method}' for integer encoding.")

            cross_fn = _INT_CROSSOVER[cross_method]
            cross_prob = float(cross_params.get("prob", 0.9))
            mut_fn = _INT_MUTATION[mut_method]
            mut_prob = resolve_prob_expression(mut_params.get("prob"), n_var, 1.0 / max(1, n_var))
            step = int(mut_params.get("step", 1))

            crossover = lambda parents, rng=rng: cross_fn(parents, cross_prob, rng)
            if mut_fn is creep_mutation:
                mutation = lambda X_child, rng=rng: mut_fn(X_child, mut_prob, step, xl, xu, rng) or X_child
            else:
                mutation = lambda X_child, rng=rng: mut_fn(X_child, mut_prob, xl, xu, rng) or X_child

        elif encoding in {"continuous", "real"}:
            cross_prob = float(cross_params.get("prob", 0.9))
            cross_eta = float(cross_params.get("eta", 20.0))
            workspace = VariationWorkspace()

            crossover_operator = SBXCrossover(
                prob_crossover=cross_prob,
                eta=cross_eta,
                lower=xl,
                upper=xu,
                workspace=workspace,
                allow_inplace=True,
            )

            mut_prob = resolve_prob_expression(mut_params.get("prob"), n_var, 1.0 / max(1, n_var))
            mut_eta = float(mut_params.get("eta", 20.0))
            mutation_operator = PolynomialMutation(
                prob_mutation=mut_prob,
                eta=mut_eta,
                lower=xl,
                upper=xu,
                workspace=workspace,
            )

            crossover = lambda parents, rng=rng: crossover_operator(parents, rng)
            mutation = lambda X_child, rng=rng: mutation_operator(X_child, rng)

        else:
            raise ValueError(f"MOEA/D does not support encoding '{encoding}'.")

        return crossover, mutation

    def ask(self) -> np.ndarray:
        """
        Generate offspring solutions to be evaluated.

        Returns
        -------
        np.ndarray
            Offspring decision variables to evaluate, shape (batch_size, n_var).

        Raises
        ------
        RuntimeError
            If called before initialization.
        """
        st = self._st
        if st is None:
            raise RuntimeError("ask() called before initialization.")

        pop_size = st.pop_size
        order = st.rng.permutation(pop_size)
        batch_size = pop_size  # Process all subproblems per generation
        active = order[:batch_size]

        # Select parent pairs
        all_indices = np.arange(pop_size)
        parent_pairs = np.empty((batch_size, 2), dtype=int)

        for pos, i in enumerate(active):
            mating_pool = st.neighbors[i] if st.rng.random() < st.delta else all_indices
            if mating_pool.size < 2:
                mating_pool = all_indices
            parent_pairs[pos] = st.rng.choice(mating_pool, size=2, replace=False)

        # Generate offspring
        parents_flat = parent_pairs.reshape(-1)
        n_var = st.X.shape[1]
        parents = st.X[parents_flat].reshape(batch_size, 2, n_var)
        offspring = st.crossover_fn(parents)
        children = offspring[:, 0, :].copy()
        children = st.mutation_fn(children)

        # Store pending info for tell()
        st.pending_offspring = children
        st.pending_active_indices = active
        st.pending_parent_pairs = parent_pairs

        return children

    def tell(self, eval_result: Any, problem: "ProblemProtocol | None" = None) -> bool:
        """
        Receive evaluated offspring and update algorithm state.

        Parameters
        ----------
        eval_result : Any
            Evaluation result with F (objectives) and optionally G (constraints).
        problem : ProblemProtocol | None
            Problem instance (optional, for constraint evaluation).

        Returns
        -------
        bool
            True if HV threshold reached.

        Raises
        ------
        RuntimeError
            If called before initialization or without pending ask().
        """
        st = self._st
        if st is None:
            raise RuntimeError("tell() called before initialization.")

        children = st.pending_offspring
        active = st.pending_active_indices

        if children is None or active is None:
            raise ValueError("tell() called without a pending ask().")

        F_child = eval_result.F
        G_child = eval_result.G if st.constraint_mode != "none" else None
        batch_size = children.shape[0]
        st.n_eval += batch_size

        # Clear pending
        st.pending_offspring = None
        st.pending_active_indices = None
        st.pending_parent_pairs = None

        # Update ideal point
        st.ideal = np.minimum(st.ideal, F_child.min(axis=0))

        # Update neighborhoods
        for pos, i in enumerate(active):
            child = children[pos]
            child_f = F_child[pos]
            child_g = G_child[pos] if G_child is not None else None
            cv_penalty = compute_violation(G_child)[pos] if G_child is not None else 0.0

            self._update_neighborhood(
                st=st,
                idx=i,
                child=child,
                child_f=child_f,
                child_g=child_g,
                cv_penalty=cv_penalty,
            )

        # Update archive
        update_archive(st)

        # Check HV termination
        hv_reached = st.hv_tracker is not None and st.hv_tracker.enabled and st.hv_tracker.reached(st.hv_points())
        return hv_reached

    def _update_neighborhood(
        self,
        st: MOEADState,
        idx: int,
        child: np.ndarray,
        child_f: np.ndarray,
        child_g: np.ndarray | None,
        cv_penalty: float,
    ) -> None:
        """Update neighborhood with a new offspring using aggregation comparison."""
        constraint_mode = st.constraint_mode
        if constraint_mode == "none" or st.G is None or child_g is None:
            constraint_mode = "none"

        neighbor_idx = st.neighbors[idx]
        if neighbor_idx.size == 0:
            return

        local_weights = st.weights[neighbor_idx]
        current_vals = st.aggregator(st.F[neighbor_idx], local_weights, st.ideal)
        child_vals = st.aggregator(child_f.reshape(1, -1), local_weights, st.ideal).ravel()

        if constraint_mode != "none":
            child_cv = cv_penalty
            current_cv = compute_violation(st.G[neighbor_idx]) if st.G is not None else np.zeros_like(current_vals)
            feas_child = child_cv <= 0.0
            feas_curr = current_cv <= 0.0

            better_mask = np.zeros_like(current_vals, dtype=bool)
            better_mask |= (~feas_curr & feas_child)
            if feas_child:
                better_mask |= (feas_curr & (child_vals < current_vals))
            else:
                better_mask |= (~feas_curr & (child_cv < current_cv))

            if not np.any(better_mask):
                return
            candidates = neighbor_idx[better_mask]
        else:
            improved_mask = child_vals < current_vals
            if not np.any(improved_mask):
                return
            candidates = neighbor_idx[improved_mask]

        if candidates.size > st.replace_limit:
            replace_idx = st.rng.choice(candidates.size, size=st.replace_limit, replace=False)
            candidates = candidates[replace_idx]

        st.X[candidates] = child
        st.F[candidates] = child_f
        if st.G is not None and child_g is not None:
            st.G[candidates] = child_g

    # =========================================================================
    # Static helpers for aggregation and neighborhoods
    # =========================================================================

    @staticmethod
    def _compute_neighbors(weights: np.ndarray, neighbor_size: int) -> np.ndarray:
        """Compute neighborhood indices based on weight vector distances."""
        dist = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=2)
        order = np.argsort(dist, axis=1)
        return order[:, :neighbor_size]

    @staticmethod
    def _build_aggregator(name: str, params: dict) -> Any:
        """Build aggregation function from name and parameters."""
        method = name.lower()
        if method in {"tchebycheff", "tchebychef", "tschebyscheff"}:
            return MOEAD._tchebycheff
        if method in {"weighted_sum", "weightedsum"}:
            return MOEAD._weighted_sum
        if method in {"penaltyboundaryintersection", "penalty_boundary_intersection", "pbi"}:
            theta = float(params.get("theta", 5.0))
            return lambda fvals, weights, ideal: MOEAD._pbi(fvals, weights, ideal, theta)
        if method in {"modifiedtchebycheff", "modified_tchebycheff"}:
            rho = float(params.get("rho", 0.001))
            return lambda fvals, weights, ideal: MOEAD._modified_tchebycheff(fvals, weights, ideal, rho)
        raise ValueError(f"Unsupported aggregation method '{name}'.")

    @staticmethod
    def _tchebycheff(fvals: np.ndarray, weights: np.ndarray, ideal: np.ndarray) -> np.ndarray:
        """Tchebycheff aggregation: max(w * |f - z*|)."""
        diff = np.abs(fvals - ideal)
        return np.max(weights * diff, axis=-1)

    @staticmethod
    def _weighted_sum(fvals: np.ndarray, weights: np.ndarray, ideal: np.ndarray) -> np.ndarray:
        """Weighted sum aggregation: sum(w * (f - z*))."""
        shifted = fvals - ideal
        return np.sum(weights * shifted, axis=-1)

    @staticmethod
    def _pbi(fvals: np.ndarray, weights: np.ndarray, ideal: np.ndarray, theta: float) -> np.ndarray:
        """Penalty boundary intersection (PBI) aggregation."""
        diff = fvals - ideal
        norm_w = np.linalg.norm(weights, axis=-1, keepdims=True)
        norm_w = np.where(norm_w > 0, norm_w, 1.0)
        w_unit = weights / norm_w
        d1 = np.abs(np.sum(diff * w_unit, axis=-1))
        proj = (d1[..., None]) * w_unit
        d2 = np.linalg.norm(diff - proj, axis=-1)
        return d1 + theta * d2

    @staticmethod
    def _modified_tchebycheff(fvals: np.ndarray, weights: np.ndarray, ideal: np.ndarray, rho: float) -> np.ndarray:
        """Modified Tchebycheff: max component plus weighted L1 term."""
        diff = np.abs(fvals - ideal)
        weighted = weights * diff
        return np.max(weighted, axis=-1) + rho * np.sum(weighted, axis=-1)


__all__ = ["MOEAD"]
