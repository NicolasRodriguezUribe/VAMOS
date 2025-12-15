"""IBEA: Indicator-Based Evolutionary Algorithm.

This module implements IBEA, which uses quality indicators (epsilon or
hypervolume) to guide selection. Fitness is computed from pairwise indicator
comparisons, promoting solutions that contribute most to overall quality.

Key Features:
    - Additive epsilon indicator for efficient computation
    - Optional hypervolume indicator for stronger selection pressure
    - Scaling factor (kappa) for fitness granularity control
    - Support for continuous encodings via VariationPipeline
    - Ask/tell interface for flexible execution
    - HV-based early termination
    - Live visualization callbacks
    - External archive support

References:
    E. Zitzler and S. Künzli, "Indicator-Based Selection in Multiobjective
    Search," in Proc. PPSN VIII, 2004, pp. 832-842.

Example:
    >>> from vamos import optimize, IBEAConfig, ZDT1
    >>> result = optimize(
    ...     problem=ZDT1(n_var=30),
    ...     algorithm="ibea",
    ...     algorithm_config=IBEAConfig().pop_size(100).indicator("epsilon").fixed(),
    ...     termination=("n_eval", 10000),
    ... )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.engine.algorithm.components.base import (
    get_eval_backend,
    get_live_viz,
    parse_termination,
    resolve_archive_size,
    setup_archive,
    setup_hv_tracker,
)
from vamos.engine.algorithm.components.hypervolume import hypervolume
from vamos.engine.algorithm.components.population import (
    evaluate_population,
    evaluate_population_with_constraints,
    initialize_population,
    resolve_bounds,
)
from vamos.engine.algorithm.components.variation import VariationPipeline, prepare_mutation_params
from vamos.engine.algorithm.ibea_state import IBEAState, build_ibea_result
from vamos.engine.algorithm.nsgaii_helpers import build_mating_pool
from vamos.engine.operators.real import VariationWorkspace
from vamos.foundation.constraints.utils import compute_violation, is_feasible

if TYPE_CHECKING:
    from vamos.foundation.eval.backends import EvaluationBackend
    from vamos.foundation.kernel.backend import KernelBackend
    from vamos.foundation.problem.types import ProblemProtocol
    from vamos.ux.visualization.live_viz import LiveVisualization


def _epsilon_indicator(F: np.ndarray) -> np.ndarray:
    """Compute additive epsilon indicator matrix."""
    diff = F[:, None, :] - F[None, :, :]
    return np.max(diff, axis=2)


def _hypervolume_indicator(F: np.ndarray) -> np.ndarray:
    """Compute hypervolume indicator matrix."""
    n = F.shape[0]
    if n == 0:
        return np.empty((0, 0))
    ref = np.max(F, axis=0) + 1.0
    indicator = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pair = np.vstack([F[i], F[j]])
            hv_pair = hypervolume(pair, ref)
            hv_j = hypervolume(F[j : j + 1], ref)
            indicator[i, j] = hv_j - hv_pair
    return indicator


def _compute_indicator_matrix(F: np.ndarray, indicator: str) -> np.ndarray:
    """Compute indicator matrix based on selected type."""
    if indicator == "hypervolume":
        return _hypervolume_indicator(F)
    return _epsilon_indicator(F)


def _ibea_fitness(indicator: np.ndarray, kappa: float) -> np.ndarray:
    """Compute IBEA fitness from indicator matrix."""
    mat = indicator.copy()
    np.fill_diagonal(mat, np.inf)
    contrib = np.exp(-mat / kappa)
    contrib[~np.isfinite(contrib)] = 0.0
    return -np.sum(contrib, axis=0)


def _apply_constraint_penalty(fitness: np.ndarray, G: np.ndarray | None) -> np.ndarray:
    """Apply constraint penalty to fitness values."""
    if G is None:
        return fitness
    cv = compute_violation(G)
    feas = is_feasible(G)
    if not feas.any():
        return fitness + cv
    penalty = np.max(np.abs(fitness)) + 1.0
    penalized = fitness.copy()
    penalized[~feas] += penalty * (1.0 + cv[~feas])
    return penalized


def _environmental_selection(
    X: np.ndarray,
    F: np.ndarray,
    G: np.ndarray | None,
    pop_size: int,
    indicator: str,
    kappa: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Perform IBEA environmental selection."""
    ind = _compute_indicator_matrix(F, indicator)
    fitness = _ibea_fitness(ind, kappa)
    fitness = _apply_constraint_penalty(fitness, G)

    while X.shape[0] > pop_size:
        worst = int(np.argmax(fitness))
        delta = np.exp(-ind[worst] / kappa)
        delta[worst] = 0.0
        fitness -= delta
        X = np.delete(X, worst, axis=0)
        F = np.delete(F, worst, axis=0)
        if G is not None:
            G = np.delete(G, worst, axis=0)
        ind = np.delete(np.delete(ind, worst, axis=0), worst, axis=1)
        fitness = np.delete(fitness, worst, axis=0)
    return X, F, G, fitness


class IBEA:
    """Indicator-Based Evolutionary Algorithm.

    IBEA uses quality indicators (epsilon or hypervolume) to compute fitness
    from pairwise comparisons. Solutions are selected based on their contribution
    to the indicator value, promoting both convergence and diversity.

    Parameters
    ----------
    config : dict
        Algorithm configuration with keys:
        - pop_size (int): Population size
        - crossover (tuple): Crossover operator config
        - mutation (tuple): Mutation operator config
        - indicator (str, optional): "epsilon" (default) or "hypervolume"
        - kappa (float, optional): Scaling factor (default: 0.05)
        - constraint_mode (str, optional): "none" or "penalty"
        - external_archive_size (int, optional): Size of external archive
        - archive_type (str, optional): Archive type ("hypervolume" or "crowding")
        - hv_threshold (float, optional): HV threshold for early termination
    kernel : KernelBackend
        Backend for vectorized operations.

    Examples
    --------
    Basic usage:

    >>> from vamos import IBEAConfig
    >>> config = IBEAConfig().pop_size(100).indicator("epsilon").kappa(0.05).fixed()
    >>> ibea = IBEA(config, kernel)
    >>> result = ibea.run(problem, ("n_eval", 10000), seed=42)

    Ask/tell interface:

    >>> ibea = IBEA(config, kernel)
    >>> ibea.initialize(problem, ("n_eval", 10000), seed=42)
    >>> while not ibea.should_terminate():
    ...     X = ibea.ask()
    ...     F = evaluate(X)
    ...     ibea.tell(X, F)
    >>> result = ibea.result()
    """

    def __init__(self, config: dict, kernel: "KernelBackend"):
        self.cfg = config
        self.kernel = kernel
        self._st: IBEAState | None = None
        self._live_cb: "LiveVisualization | None" = None
        self._eval_backend: "EvaluationBackend | None" = None
        self._max_eval: int = 0
        self._hv_tracker: Any = None
        self._problem: "ProblemProtocol | None" = None

    # -------------------------------------------------------------------------
    # Main run method (batch mode)
    # -------------------------------------------------------------------------

    def run(
        self,
        problem: "ProblemProtocol",
        termination: tuple[str, Any],
        seed: int,
        eval_backend: "EvaluationBackend | None" = None,
        live_viz: "LiveVisualization | None" = None,
    ) -> dict[str, Any]:
        """Run IBEA optimization loop.

        Parameters
        ----------
        problem : ProblemProtocol
            Problem to optimize.
        termination : tuple
            Termination criterion, e.g., ("n_eval", 10000).
        seed : int
            Random seed for reproducibility.
        eval_backend : EvaluationBackend, optional
            Evaluation backend for parallel evaluation.
        live_viz : LiveVisualization, optional
            Live visualization callback.

        Returns
        -------
        dict
            Result dictionary with X, F, G, archive data.
        """
        live_cb, eval_backend, max_eval, hv_tracker = self._initialize_run(
            problem, termination, seed, eval_backend, live_viz
        )
        self._problem = problem

        st = self._st
        if st is None:
            raise RuntimeError("State not initialized")

        hv_reached = False
        while st.n_eval < max_eval:
            # Generate offspring
            X_off = self._generate_offspring(st)

            # Evaluate offspring
            F_off, G_off = self._evaluate_offspring(
                problem, X_off, eval_backend, st.constraint_mode
            )
            st.n_eval += X_off.shape[0]

            # Environmental selection
            X_comb = np.vstack([st.X, X_off])
            F_comb = np.vstack([st.F, F_off])
            G_comb = self._combine_constraints(st.G, G_off)

            st.X, st.F, st.G, st.fitness = _environmental_selection(
                X_comb, F_comb, G_comb, st.pop_size, st.indicator, st.kappa
            )

            st.generation += 1

            # Update archive
            if st.archive_manager is not None:
                st.archive_manager.update(st.X, st.F)
                st.archive_X, st.archive_F = st.archive_manager.get_archive()

            # Live callback
            live_cb.on_generation(st.generation, F=st.F)

            # Check HV threshold
            if hv_tracker is not None:
                hv_tracker.update(st.F)
                if hv_tracker.reached_threshold():
                    hv_reached = True
                    break

        live_cb.on_end(final_F=st.F)
        return build_ibea_result(st, hv_reached)

    # -------------------------------------------------------------------------
    # Initialization helpers
    # -------------------------------------------------------------------------

    def _initialize_run(
        self,
        problem: "ProblemProtocol",
        termination: tuple[str, Any],
        seed: int,
        eval_backend: "EvaluationBackend | None" = None,
        live_viz: "LiveVisualization | None" = None,
    ) -> tuple[Any, Any, int, Any]:
        """Initialize the algorithm run."""
        max_eval, hv_config = parse_termination(termination, self.cfg)

        live_cb = get_live_viz(live_viz)
        eval_backend = get_eval_backend(eval_backend)

        # Setup HV tracker if configured
        hv_tracker = None
        if hv_config is not None:
            hv_tracker = setup_hv_tracker(hv_config, problem.n_obj)

        rng = np.random.default_rng(seed)
        pop_size = int(self.cfg["pop_size"])
        offspring_size = pop_size
        encoding = getattr(problem, "encoding", "continuous")
        n_var = problem.n_var
        n_obj = problem.n_obj
        xl, xu = resolve_bounds(problem, encoding)
        constraint_mode = self.cfg.get("constraint_mode", "none")

        # Initialize population
        initializer_cfg = self.cfg.get("initializer")
        X = initialize_population(pop_size, n_var, xl, xu, encoding, rng, problem, initializer=initializer_cfg)

        if constraint_mode and constraint_mode != "none":
            F, G = evaluate_population_with_constraints(problem, X)
        else:
            F = evaluate_population(problem, X)
            G = None
        n_eval = X.shape[0]

        # Selection pressure
        sel_method, sel_params = self.cfg["selection"]
        pressure = int(sel_params.get("pressure", 2))

        # Build variation pipeline
        cross_method, cross_params = self.cfg["crossover"]
        cross_method = cross_method.lower()
        cross_params = dict(cross_params)

        mut_method, mut_params = self.cfg["mutation"]
        mut_method = mut_method.lower()
        mut_params = prepare_mutation_params(
            mut_params, encoding, n_var, prob_factor=self.cfg.get("mutation_prob_factor")
        )

        variation_workspace = VariationWorkspace()
        variation = VariationPipeline(
            encoding=encoding,
            cross_method=cross_method,
            cross_params=cross_params,
            mut_method=mut_method,
            mut_params=mut_params,
            xl=xl,
            xu=xu,
            workspace=variation_workspace,
            repair_cfg=self.cfg.get("repair"),
            problem=problem,
        )

        # IBEA parameters
        indicator = self.cfg.get("indicator", "eps").lower()
        if indicator in ("eps", "epsilon", "additive_epsilon"):
            indicator = "epsilon"
        kappa = float(self.cfg.get("kappa", 0.05))

        # Compute initial fitness
        _, _, _, fitness = _environmental_selection(
            X.copy(), F.copy(), G.copy() if G is not None else None,
            pop_size, indicator, kappa
        )

        # Setup external archive
        archive_size = resolve_archive_size(self.cfg) or 0
        archive_manager = None
        archive_X: np.ndarray | None = None
        archive_F: np.ndarray | None = None

        if archive_size > 0:
            archive_type = self.cfg.get("archive_type", "hypervolume")
            archive_manager, archive_X, archive_F = setup_archive(
                kernel=self.kernel,
                X=X,
                F=F,
                n_var=n_var,
                n_obj=n_obj,
                dtype=X.dtype,
                archive_size=archive_size,
                archive_type=archive_type,
            )

        live_cb.on_start(problem=problem, algorithm=self, config=self.cfg)

        # Create state
        self._st = IBEAState(
            X=X,
            F=F,
            G=G,
            rng=rng,
            pop_size=pop_size,
            offspring_size=offspring_size,
            constraint_mode=constraint_mode,
            n_eval=n_eval,
            # IBEA specific
            indicator=indicator,
            kappa=kappa,
            fitness=fitness,
            pressure=pressure,
            variation=variation,
            # Archive
            archive_size=archive_size,
            archive_X=archive_X,
            archive_F=archive_F,
            archive_manager=archive_manager,
            # Termination
            hv_tracker=hv_tracker,
        )

        return live_cb, eval_backend, max_eval, hv_tracker

    def _generate_offspring(self, st: IBEAState) -> np.ndarray:
        """Generate offspring using tournament selection and variation."""
        ranks = np.argsort(np.argsort(st.fitness))
        crowd = np.zeros_like(st.fitness, dtype=float)
        parents_per_group = st.variation.parents_per_group
        children_per_group = st.variation.children_per_group
        parent_count = int(np.ceil(st.offspring_size / children_per_group) * parents_per_group)

        sel_method, _ = self.cfg["selection"]
        mating_pairs = build_mating_pool(
            self.kernel, ranks, crowd, st.pressure, st.rng, parent_count, parents_per_group, sel_method
        )
        parent_idx = mating_pairs.reshape(-1)
        X_parents = st.variation.gather_parents(st.X, parent_idx)
        X_off = st.variation.produce_offspring(X_parents, st.rng)
        if X_off.shape[0] > st.offspring_size:
            X_off = X_off[:st.offspring_size]

        return X_off

    def _evaluate_offspring(
        self,
        problem: "ProblemProtocol",
        X: np.ndarray,
        eval_backend: "EvaluationBackend",
        constraint_mode: str,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Evaluate offspring and compute constraints."""
        if constraint_mode and constraint_mode != "none":
            F, G = evaluate_population_with_constraints(problem, X)
        else:
            F = evaluate_population(problem, X)
            G = None
        return F, G

    @staticmethod
    def _combine_constraints(
        G: np.ndarray | None, G_off: np.ndarray | None
    ) -> np.ndarray | None:
        """Combine parent and offspring constraints."""
        if G is None and G_off is None:
            return None
        if G is None:
            return G_off
        if G_off is None:
            return G
        return np.vstack([G, G_off])

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
            self._st.pending_offspring = None

    def ask(self) -> np.ndarray:
        """Generate offspring for evaluation.

        Returns
        -------
        np.ndarray
            Offspring decision vectors to evaluate.
        """
        if self._st is None:
            raise RuntimeError("Algorithm not initialized. Call initialize() first.")
        if self._st.pending_offspring is not None:
            raise RuntimeError("Previous offspring not yet consumed by tell().")

        offspring = self._generate_offspring(self._st)
        self._st.pending_offspring = offspring
        return offspring.copy()

    def tell(
        self,
        X: np.ndarray,
        F: np.ndarray,
        G: np.ndarray | None = None,
    ) -> None:
        """Receive evaluated offspring and update population.

        Parameters
        ----------
        X : np.ndarray
            Evaluated decision vectors.
        F : np.ndarray
            Objective values.
        G : np.ndarray, optional
            Constraint values.
        """
        if self._st is None:
            raise RuntimeError("Algorithm not initialized. Call initialize() first.")
        if self._st.pending_offspring is None:
            raise RuntimeError("No pending offspring. Call ask() first.")

        st = self._st
        st.pending_offspring = None

        # Environmental selection
        X_comb = np.vstack([st.X, X])
        F_comb = np.vstack([st.F, F])
        G_comb = self._combine_constraints(st.G, G)

        st.X, st.F, st.G, st.fitness = _environmental_selection(
            X_comb, F_comb, G_comb, st.pop_size, st.indicator, st.kappa
        )

        st.n_eval += X.shape[0]
        st.generation += 1

        # Update archive
        if st.archive_manager is not None:
            st.archive_manager.update(st.X, st.F)
            st.archive_X, st.archive_F = st.archive_manager.get_archive()

        # Live callback
        if self._live_cb is not None:
            self._live_cb.on_generation(st.generation, F=st.F)

        # Check HV tracker
        if st.hv_tracker is not None:
            st.hv_tracker.update(st.F)

    def should_terminate(self) -> bool:
        """Check if termination criterion is met."""
        if self._st is None:
            return True
        if self._st.n_eval >= self._max_eval:
            return True
        if self._st.hv_tracker is not None and self._st.hv_tracker.reached_threshold():
            return True
        return False

    def result(self) -> dict[str, Any]:
        """Get current result."""
        if self._st is None:
            raise RuntimeError("Algorithm not initialized.")

        hv_reached = (
            self._st.hv_tracker is not None
            and self._st.hv_tracker.reached_threshold()
        )

        if self._live_cb is not None:
            self._live_cb.on_end(final_F=self._st.F)

        return build_ibea_result(self._st, hv_reached)

    @property
    def state(self) -> IBEAState | None:
        """Access current algorithm state."""
        return self._st
