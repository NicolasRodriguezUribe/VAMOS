"""SMS-EMOA: S-Metric Selection Evolutionary Multiobjective Optimization Algorithm.

This module implements SMS-EMOA, which uses the hypervolume indicator (S-metric)
for survival selection. Each generation, solutions with the smallest hypervolume
contribution to the current Pareto front are removed.

Key Features:
    - Hypervolume-based selection for convergence and diversity
    - Support for continuous, binary, and integer encodings
    - Dynamic reference point computation
    - Strong theoretical convergence guarantees
    - Ask/tell interface for flexible execution
    - HV-based early termination
    - Live visualization callbacks
    - External archive support

References:
    N. Beume, B. Naujoks, and M. Emmerich, "SMS-EMOA: Multiobjective Selection
    Based on Dominated Hypervolume," European Journal of Operational Research,
    vol. 181, no. 3, 2007.

Example:
    >>> from vamos import optimize, SMSEMOAConfig, ZDT1
    >>> result = optimize(
    ...     problem=ZDT1(n_var=30),
    ...     algorithm="smsemoa",
    ...     algorithm_config=SMSEMOAConfig().pop_size(100).fixed(),
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
from vamos.engine.algorithm.components.hypervolume import hypervolume_contributions
from vamos.engine.algorithm.components.utils import (
    resolve_bounds_array,
    resolve_prob_expression,
)
from vamos.engine.algorithm.smsemoa_state import SMSEMOAState, build_smsemoa_result
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
    from vamos.foundation.problem.protocol import ProblemProtocol
    from vamos.foundation.kernel.protocols import KernelBackend
    from vamos.engine.algorithm.components.base import EvaluationBackend, LiveVisualization


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


class SMSEMOA:
    """S-Metric Selection Evolutionary Multiobjective Optimization Algorithm.

    SMS-EMOA uses hypervolume contribution for survival selection, removing
    solutions that contribute least to the hypervolume indicator each generation.
    This provides strong convergence toward the Pareto front.

    Parameters
    ----------
    config : dict
        Algorithm configuration with keys:
        - pop_size (int): Population size
        - crossover (tuple): Crossover operator config
        - mutation (tuple): Mutation operator config
        - selection (tuple): Selection config, e.g., ("tournament", {"pressure": 2})
        - reference_point (dict, optional): {"offset": float} or {"point": list}
        - external_archive_size (int, optional): Size of external archive
        - archive_type (str, optional): Archive type ("hypervolume" or "crowding")
        - hv_threshold (float, optional): HV threshold for early termination
        - hv_ref_point (list, optional): Reference point for HV computation
    kernel : KernelBackend
        Backend for vectorized operations.

    Examples
    --------
    Basic usage:

    >>> from vamos import SMSEMOAConfig
    >>> config = SMSEMOAConfig().pop_size(100).crossover("sbx", prob=0.9).fixed()
    >>> smsemoa = SMSEMOA(config, kernel)
    >>> result = smsemoa.run(problem, ("n_eval", 10000), seed=42)

    Ask/tell interface:

    >>> smsemoa = SMSEMOA(config, kernel)
    >>> smsemoa.initialize(problem, ("n_eval", 10000), seed=42)
    >>> while not smsemoa.should_terminate():
    ...     X = smsemoa.ask()
    ...     F = evaluate(X)
    ...     smsemoa.tell(X, F)
    >>> result = smsemoa.result()
    """

    def __init__(self, config: dict, kernel: "KernelBackend"):
        self.cfg = config
        self.kernel = kernel
        self._st: SMSEMOAState | None = None
        self._live_cb: LiveVisualization | None = None
        self._eval_backend: EvaluationBackend | None = None
        self._max_eval: int = 0
        self._hv_tracker: Any = None
        self._problem: ProblemProtocol | None = None

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
        """Run SMS-EMOA optimization loop.

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
            Result dictionary with X, F, G, reference_point, archive data.
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
            # Generate and evaluate offspring
            X_child = self._generate_offspring(st)

            # Evaluate using backend or directly
            F_child, G_child = self._evaluate_offspring(
                problem, X_child, eval_backend, st.constraint_mode
            )
            st.n_eval += X_child.shape[0]

            # Survival selection (one child at a time for SMS-EMOA)
            for i in range(X_child.shape[0]):
                self._survival_selection(
                    st, X_child[i:i+1], F_child[i:i+1],
                    G_child[i:i+1] if G_child is not None else None
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
        return build_smsemoa_result(st, hv_reached)

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
        """Initialize the algorithm run.

        Returns
        -------
        tuple
            (live_cb, eval_backend, max_eval, hv_tracker)
        """
        max_eval, hv_config = parse_termination(termination, self.cfg)

        live_cb = get_live_viz(live_viz)
        eval_backend = get_eval_backend(eval_backend)

        # Setup HV tracker if configured
        hv_tracker = None
        if hv_config is not None:
            hv_tracker = setup_hv_tracker(hv_config, problem.n_obj)

        rng = np.random.default_rng(seed)
        pop_size = self.cfg["pop_size"]
        encoding = getattr(problem, "encoding", "continuous")
        xl, xu = resolve_bounds_array(problem, encoding)
        n_var = problem.n_var
        n_obj = problem.n_obj
        constraint_mode = self.cfg.get("constraint_mode", "penalty")

        # Build variation operators
        crossover_fn, mutation_fn = self._build_variation(encoding, n_var, xl, xu, rng)

        # Selection pressure
        sel_method, sel_params = self.cfg["selection"]
        pressure = sel_params.get("pressure", 2) if sel_method == "tournament" else 2

        # Reference point config
        ref_cfg = self.cfg.get("reference_point", {}) or {}

        # Initialize population
        X, F, G = self._initialize_population(
            encoding, pop_size, n_var, xl, xu, rng, problem, constraint_mode
        )
        n_eval = pop_size

        # Initialize reference point
        ref_point, ref_offset, ref_adaptive = self._initialize_reference_point(F, ref_cfg)

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
        self._st = SMSEMOAState(
            X=X,
            F=F,
            G=G,
            rng=rng,
            pop_size=pop_size,
            offspring_size=1,  # SMS-EMOA typically generates 1 offspring per iteration
            constraint_mode=constraint_mode,
            n_eval=n_eval,
            # SMSEMOA-specific
            ref_point=ref_point,
            ref_offset=ref_offset,
            ref_adaptive=ref_adaptive,
            pressure=pressure,
            crossover_fn=crossover_fn,
            mutation_fn=mutation_fn,
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

        F, G = self._evaluate_population_with_constraints(problem, X)
        if constraint_mode == "none":
            G = None
        return X, F, G

    def _evaluate_population_with_constraints(
        self,
        problem: "ProblemProtocol",
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Evaluate population and compute constraints if present."""
        n_obj = problem.n_obj
        n_con = getattr(problem, "n_con", 0) or 0

        F = np.empty((X.shape[0], n_obj), dtype=np.float64)

        if n_con > 0:
            G = np.empty((X.shape[0], n_con), dtype=np.float64)
            problem.evaluate(X, {"F": F, "G": G})
        else:
            problem.evaluate(X, {"F": F})
            G = None

        return F, G

    @staticmethod
    def _initialize_reference_point(
        F: np.ndarray,
        ref_cfg: dict,
    ) -> tuple[np.ndarray, float, bool]:
        """Initialize reference point for HV computation."""
        offset = float(ref_cfg.get("offset", 0.1))
        adaptive = bool(ref_cfg.get("adaptive", True))
        vector = ref_cfg.get("vector")

        if vector is not None:
            ref = np.asarray(vector, dtype=float)
            if ref.shape[0] != F.shape[1]:
                raise ValueError("reference_point vector dimensionality mismatch.")
            ref = np.maximum(ref, F.max(axis=0) + offset)
        else:
            ref = F.max(axis=0) + offset

        return ref, offset, adaptive

    def _build_variation(
        self,
        encoding: str,
        n_var: int,
        xl: np.ndarray,
        xu: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[Any, Any]:
        """Build crossover and mutation operators for the encoding."""
        # Unpack crossover config (format: ("sbx", {"prob": 0.9, "eta": 20.0}))
        cross_cfg = self.cfg.get("crossover", ("sbx", {}))
        if isinstance(cross_cfg, tuple):
            cross_method, cross_params = cross_cfg
            cross_params = dict(cross_params) if cross_params else {}
        else:
            cross_method = "sbx"
            cross_params = cross_cfg or {}

        # Unpack mutation config (format: ("pm", {"prob": "1/n", "eta": 20.0}))
        mut_cfg = self.cfg.get("mutation", ("pm", {}))
        if isinstance(mut_cfg, tuple):
            mut_method, mut_params = mut_cfg
            mut_params = dict(mut_params) if mut_params else {}
        else:
            mut_method = "pm"
            mut_params = mut_cfg or {}

        if mut_params.get("prob") == "1/n":
            mut_params["prob"] = 1.0 / n_var

        if encoding == "binary":
            if cross_method not in _BINARY_CROSSOVER:
                raise ValueError(f"Unsupported SMSEMOA crossover '{cross_method}' for binary encoding.")
            if mut_method not in _BINARY_MUTATION:
                raise ValueError(f"Unsupported SMSEMOA mutation '{mut_method}' for binary encoding.")

            cross_fn = _BINARY_CROSSOVER[cross_method]
            cross_prob = float(cross_params.get("prob", 0.9))
            mut_fn = _BINARY_MUTATION[mut_method]
            mut_prob = resolve_prob_expression(mut_params.get("prob"), n_var, 1.0 / max(1, n_var))

            crossover = lambda parents, rng=rng: cross_fn(parents, cross_prob, rng)
            mutation = lambda X_child, rng=rng: (mut_fn(X_child, mut_prob, rng) or X_child)

        elif encoding == "integer":
            if cross_method not in _INT_CROSSOVER:
                raise ValueError(f"Unsupported SMSEMOA crossover '{cross_method}' for integer encoding.")
            if mut_method not in _INT_MUTATION:
                raise ValueError(f"Unsupported SMSEMOA mutation '{mut_method}' for integer encoding.")

            cross_fn = _INT_CROSSOVER[cross_method]
            cross_prob = float(cross_params.get("prob", 0.9))
            mut_fn = _INT_MUTATION[mut_method]
            mut_prob = resolve_prob_expression(mut_params.get("prob"), n_var, 1.0 / max(1, n_var))
            step = int(mut_params.get("step", 1))

            crossover = lambda parents, rng=rng: cross_fn(parents, cross_prob, rng)
            if mut_fn is creep_mutation:
                mutation = lambda X_child, rng=rng: (mut_fn(X_child, mut_prob, step, xl, xu, rng) or X_child)
            else:
                mutation = lambda X_child, rng=rng: (mut_fn(X_child, mut_prob, xl, xu, rng) or X_child)

        elif encoding in {"continuous", "real"}:
            cross_prob = float(cross_params.get("prob", 0.9))
            cross_eta = float(cross_params.get("eta", 20.0))
            workspace = VariationWorkspace()

            sbx = SBXCrossover(
                prob_crossover=cross_prob,
                eta=cross_eta,
                lower=xl,
                upper=xu,
                workspace=workspace,
                allow_inplace=True,
            )

            mut_prob = resolve_prob_expression(mut_params.get("prob"), n_var, 1.0 / max(1, n_var))
            mut_eta = float(mut_params.get("eta", 20.0))
            pm = PolynomialMutation(
                prob_mutation=mut_prob,
                eta=mut_eta,
                lower=xl,
                upper=xu,
                workspace=workspace,
            )

            crossover = lambda parents, rng=rng: sbx(parents, rng)
            mutation = lambda X_child, rng=rng: pm(X_child, rng)

        else:
            raise ValueError(f"SMSEMOA does not support encoding '{encoding}'.")

        return crossover, mutation

    # -------------------------------------------------------------------------
    # Generation logic
    # -------------------------------------------------------------------------

    def _generate_offspring(self, st: SMSEMOAState) -> np.ndarray:
        """Generate offspring using tournament selection and variation."""
        # Tournament selection for parent indices
        ranks, crowd = self.kernel.nsga2_ranking(st.F)
        parents_idx = self.kernel.tournament_selection(
            ranks, crowd, st.pressure, st.rng, n_parents=2
        )

        parents = st.X[parents_idx]
        if parents.ndim == 2:
            parents = parents.reshape(1, 2, -1)

        # Apply crossover and mutation
        offspring = st.crossover_fn(parents)
        child_vec = offspring.reshape(-1, st.X.shape[1])[0:1]  # first child as (1, n_var)
        child = st.mutation_fn(child_vec)

        return child

    def _evaluate_offspring(
        self,
        problem: "ProblemProtocol",
        X: np.ndarray,
        eval_backend: "EvaluationBackend",
        constraint_mode: str,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Evaluate offspring and compute constraints."""
        n_obj = problem.n_obj
        n_con = getattr(problem, "n_con", 0) or 0

        F = np.empty((X.shape[0], n_obj), dtype=np.float64)

        if n_con > 0:
            G = np.empty((X.shape[0], n_con), dtype=np.float64)
            problem.evaluate(X, {"F": F, "G": G})
        else:
            problem.evaluate(X, {"F": F})
            G = None

        if constraint_mode == "none":
            G = None

        return F, G

    def _survival_selection(
        self,
        st: SMSEMOAState,
        X_child: np.ndarray,
        F_child: np.ndarray,
        G_child: np.ndarray | None,
    ) -> None:
        """Perform survival selection, removing worst HV contributor."""
        # Combine population with child
        X_comb = np.vstack([st.X, X_child])
        F_comb = np.vstack([st.F, F_child])
        G_comb = np.vstack([st.G, G_child]) if st.G is not None and G_child is not None else None

        # Update reference point if adaptive
        if st.ref_adaptive:
            st.ref_point = np.maximum(st.ref_point, F_child[0] + st.ref_offset)

        # Non-dominated ranking
        ranks, _ = self.kernel.nsga2_ranking(F_comb)
        worst_rank = ranks.max()
        worst_idx = np.flatnonzero(ranks == worst_rank)

        if worst_idx.size == 1:
            remove_idx = worst_idx[0]
        else:
            # Remove solution with smallest HV contribution
            contribs = hypervolume_contributions(F_comb[worst_idx], st.ref_point)
            remove_idx = worst_idx[np.argmin(contribs)]

        # Keep all except removed
        keep = np.delete(np.arange(F_comb.shape[0]), remove_idx)
        st.X = X_comb[keep][:st.pop_size]
        st.F = F_comb[keep][:st.pop_size]
        if G_comb is not None:
            st.G = G_comb[keep][:st.pop_size]

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
        """Initialize algorithm for ask/tell loop.

        Parameters
        ----------
        problem : ProblemProtocol
            Problem to optimize.
        termination : tuple
            Termination criterion.
        seed : int
            Random seed.
        eval_backend : EvaluationBackend, optional
            Evaluation backend.
        live_viz : LiveVisualization, optional
            Live visualization callback.
        """
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

        Raises
        ------
        RuntimeError
            If algorithm not initialized or previous offspring not consumed.
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

        Raises
        ------
        RuntimeError
            If algorithm not initialized or no pending offspring.
        """
        if self._st is None:
            raise RuntimeError("Algorithm not initialized. Call initialize() first.")
        if self._st.pending_offspring is None:
            raise RuntimeError("No pending offspring. Call ask() first.")

        st = self._st
        st.pending_offspring = None

        # Survival selection for each child
        for i in range(X.shape[0]):
            G_i = G[i:i+1] if G is not None else None
            self._survival_selection(st, X[i:i+1], F[i:i+1], G_i)

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
        """Check if termination criterion is met.

        Returns
        -------
        bool
            True if algorithm should stop.
        """
        if self._st is None:
            return True
        if self._st.n_eval >= self._max_eval:
            return True
        if self._st.hv_tracker is not None and self._st.hv_tracker.reached_threshold():
            return True
        return False

    def result(self) -> dict[str, Any]:
        """Get current result.

        Returns
        -------
        dict
            Result dictionary with X, F, G, reference_point, archive data.

        Raises
        ------
        RuntimeError
            If algorithm not initialized.
        """
        if self._st is None:
            raise RuntimeError("Algorithm not initialized.")

        hv_reached = (
            self._st.hv_tracker is not None
            and self._st.hv_tracker.reached_threshold()
        )

        if self._live_cb is not None:
            self._live_cb.on_end(final_F=self._st.F)

        return build_smsemoa_result(self._st, hv_reached)

    @property
    def state(self) -> SMSEMOAState | None:
        """Access current algorithm state."""
        return self._st
