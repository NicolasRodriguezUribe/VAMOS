"""NSGA-III: Non-dominated Sorting Genetic Algorithm III.

This module implements NSGA-III, designed for many-objective optimization
(problems with 3+ objectives). It uses reference points to maintain diversity
across the Pareto front and employs a niching mechanism for survival selection.

Key Features:
    - Reference-point based diversity preservation
    - Das-Dennis simplex-lattice reference point generation
    - Support for continuous, binary, and integer encodings
    - Scalable to many-objective problems (>3 objectives)
    - Ask/tell interface for flexible execution
    - HV-based early termination
    - Live visualization callbacks
    - External archive support

References:
    K. Deb and H. Jain, "An Evolutionary Many-Objective Optimization Algorithm
    Using Reference-Point-Based Nondominated Sorting Approach, Part I: Solving
    Problems With Box Constraints," IEEE Trans. Evolutionary Computation,
    vol. 18, no. 4, 2014.

Example:
    >>> from vamos import optimize, NSGA3Config, DTLZ1
    >>> result = optimize(
    ...     problem=DTLZ1(n_var=7, n_obj=3),
    ...     algorithm="nsga3",
    ...     algorithm_config=NSGA3Config().pop_size(92).fixed(),
    ...     termination=("n_eval", 20000),
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
from vamos.engine.algorithm.components.utils import resolve_bounds_array, resolve_prob_expression
from vamos.engine.algorithm.components.weight_vectors import load_or_generate_weight_vectors
from vamos.engine.algorithm.nsga3_state import NSGA3State, build_nsga3_result
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


class NSGAIII:
    """Non-dominated Sorting Genetic Algorithm III for many-objective optimization.

    NSGA-III uses reference points for diversity maintenance, making it suitable
    for problems with 3 or more objectives where crowding distance is less effective.
    Reference points guide the search toward a well-distributed Pareto front.

    Parameters
    ----------
    config : dict
        Algorithm configuration with keys:
        - pop_size (int): Population size (should align with reference points)
        - crossover (tuple): Crossover operator config
        - mutation (tuple): Mutation operator config
        - reference_directions (dict, optional): Reference point configuration
        - selection (tuple, optional): Parent selection config
        - external_archive_size (int, optional): Size of external archive
        - archive_type (str, optional): Archive type ("hypervolume" or "crowding")
        - hv_threshold (float, optional): HV threshold for early termination
    kernel : KernelBackend
        Backend for vectorized operations.

    Examples
    --------
    Basic usage:

    >>> from vamos import NSGA3Config
    >>> config = NSGA3Config().pop_size(92).divisions(12).fixed()
    >>> nsga3 = NSGAIII(config, kernel)
    >>> result = nsga3.run(problem, ("n_eval", 20000), seed=42)

    Ask/tell interface:

    >>> nsga3 = NSGAIII(config, kernel)
    >>> nsga3.initialize(problem, ("n_eval", 20000), seed=42)
    >>> while not nsga3.should_terminate():
    ...     X = nsga3.ask()
    ...     F = evaluate(X)
    ...     nsga3.tell(X, F)
    >>> result = nsga3.result()
    """

    def __init__(self, config: dict, kernel: "KernelBackend"):
        self.cfg = config
        self.kernel = kernel
        self._st: NSGA3State | None = None
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
        """Run NSGA-III optimization loop.

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
            Result dictionary with X, F, G, reference_directions, archive data.
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
            X_off = self._generate_offspring(st)

            # Evaluate using backend or directly
            F_off, G_off = self._evaluate_offspring(
                problem, X_off, eval_backend, st.constraint_mode
            )
            st.n_eval += X_off.shape[0]

            # NSGA-III survival selection
            st.X, st.F, st.G = self._nsga3_survival(
                st.X, st.F, st.G, X_off, F_off, G_off,
                st.pop_size, st.ref_dirs_norm, st.rng
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
        return build_nsga3_result(st, hv_reached)

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

        # Load reference directions
        dir_cfg = self.cfg.get("reference_directions", {}) or {}
        ref_dirs = load_or_generate_weight_vectors(
            pop_size, n_obj, path=dir_cfg.get("path"), divisions=dir_cfg.get("divisions")
        )
        if ref_dirs.shape[0] > pop_size:
            ref_dirs = ref_dirs[:pop_size]

        ref_dirs = np.asarray(ref_dirs, dtype=float)
        ref_dirs_norm = ref_dirs / np.linalg.norm(ref_dirs, axis=1, keepdims=True)
        ref_dirs_norm[np.isnan(ref_dirs_norm)] = 0.0

        # Initialize population
        X, F, G = self._initialize_population(
            encoding, pop_size, n_var, xl, xu, rng, problem, constraint_mode
        )
        n_eval = pop_size

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
        self._st = NSGA3State(
            X=X,
            F=F,
            G=G,
            rng=rng,
            pop_size=pop_size,
            offspring_size=pop_size,
            constraint_mode=constraint_mode,
            n_eval=n_eval,
            # NSGA-III specific
            ref_dirs=ref_dirs,
            ref_dirs_norm=ref_dirs_norm,
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

    def _build_variation(
        self,
        encoding: str,
        n_var: int,
        xl: np.ndarray,
        xu: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[Any, Any]:
        """Build crossover and mutation operators for the encoding."""
        # Unpack crossover config
        cross_cfg = self.cfg.get("crossover", ("sbx", {}))
        if isinstance(cross_cfg, tuple):
            cross_method, cross_params = cross_cfg
            cross_params = dict(cross_params) if cross_params else {}
        else:
            cross_method = "sbx"
            cross_params = cross_cfg or {}

        # Unpack mutation config
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
                raise ValueError(f"Unsupported NSGA-III crossover '{cross_method}' for binary encoding.")
            if mut_method not in _BINARY_MUTATION:
                raise ValueError(f"Unsupported NSGA-III mutation '{mut_method}' for binary encoding.")

            cross_fn = _BINARY_CROSSOVER[cross_method]
            cross_prob = float(cross_params.get("prob", 0.9))
            mut_fn = _BINARY_MUTATION[mut_method]
            mut_prob = resolve_prob_expression(mut_params.get("prob"), n_var, 1.0 / max(1, n_var))

            crossover = lambda parents, rng=rng: cross_fn(parents, cross_prob, rng)
            mutation = lambda X_child, rng=rng: (mut_fn(X_child, mut_prob, rng) or X_child)

        elif encoding == "integer":
            if cross_method not in _INT_CROSSOVER:
                raise ValueError(f"Unsupported NSGA-III crossover '{cross_method}' for integer encoding.")
            if mut_method not in _INT_MUTATION:
                raise ValueError(f"Unsupported NSGA-III mutation '{mut_method}' for integer encoding.")

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
            raise ValueError(f"NSGA-III does not support encoding '{encoding}'.")

        return crossover, mutation

    # -------------------------------------------------------------------------
    # Generation logic
    # -------------------------------------------------------------------------

    def _generate_offspring(self, st: NSGA3State) -> np.ndarray:
        """Generate offspring using tournament selection and variation."""
        n_var = st.X.shape[1]
        n_parents = 2 * (st.pop_size // 2)

        ranks, crowd = self.kernel.nsga2_ranking(st.F)
        parents_idx = self.kernel.tournament_selection(
            ranks, crowd, st.pressure, st.rng, n_parents=n_parents
        )

        X_parents = st.X[parents_idx].reshape(-1, 2, n_var)
        offspring_pairs = st.crossover_fn(X_parents)
        X_off = offspring_pairs.reshape(-1, n_var)
        X_off = st.mutation_fn(X_off)

        return X_off

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

    def _nsga3_survival(
        self,
        X: np.ndarray,
        F: np.ndarray,
        G: np.ndarray | None,
        X_off: np.ndarray,
        F_off: np.ndarray,
        G_off: np.ndarray | None,
        pop_size: int,
        ref_dirs_norm: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Perform NSGA-III survival selection with niching."""
        X_all = np.vstack([X, X_off])
        F_all = np.vstack([F, F_off])
        G_all = np.vstack([G, G_off]) if G is not None and G_off is not None else None

        fronts = self._fast_non_dominated_sort(F_all)
        new_X = []
        new_F = []
        new_G = [] if G_all is not None else None

        ideal = F_all.min(axis=0)
        shifted = F_all - ideal
        extreme_idx = self._identify_extremes(shifted)
        intercepts = self._compute_intercepts(shifted, extreme_idx)
        denom = np.where(intercepts > 0, intercepts, 1.0)
        normalized = shifted / denom

        associations, distances = self._associate(normalized, ref_dirs_norm)
        niche_counts = np.zeros(ref_dirs_norm.shape[0], dtype=int)

        for front in fronts:
            front = np.asarray(front, dtype=int)
            if len(new_X) + front.size <= pop_size:
                new_X.extend(X_all[front])
                new_F.extend(F_all[front])
                if new_G is not None:
                    new_G.extend(G_all[front])
                for idx in front:
                    niche_counts[associations[idx]] += 1
            else:
                remaining = pop_size - len(new_X)
                selected_idx = self._niche_selection(
                    front, remaining, niche_counts, associations, distances, rng
                )
                new_X.extend(X_all[selected_idx])
                new_F.extend(F_all[selected_idx])
                if new_G is not None:
                    new_G.extend(G_all[selected_idx])
                break

        return (
            np.asarray(new_X),
            np.asarray(new_F),
            np.asarray(new_G) if new_G is not None else None,
        )

    @staticmethod
    def _identify_extremes(shifted: np.ndarray) -> np.ndarray:
        """Identify extreme points using ASF (Achievement Scalarization Function)."""
        if shifted.size == 0:
            return np.array([], dtype=int)
        n_obj = shifted.shape[1]
        extremes = np.empty(n_obj, dtype=int)
        unit = np.eye(n_obj)
        for i in range(n_obj):
            weights = np.where(unit[i] == 0, 1e6, 1.0)
            asf = (shifted * weights).max(axis=1)
            extremes[i] = int(np.argmin(asf))
        return extremes

    @staticmethod
    def _compute_intercepts(shifted: np.ndarray, extreme_idx: np.ndarray) -> np.ndarray:
        """Compute intercepts from extreme points; fall back to axis-wise maxima."""
        n_obj = shifted.shape[1]
        if extreme_idx.size == 0:
            return np.ones(n_obj, dtype=float)
        extreme_pts = shifted[extreme_idx]
        intercepts = np.zeros(n_obj, dtype=float)
        try:
            b = np.ones(n_obj)
            plane = np.linalg.solve(extreme_pts, b)
            intercepts = 1.0 / plane
        except Exception:
            intercepts = shifted.max(axis=0)
        if np.any(~np.isfinite(intercepts)) or np.any(intercepts <= 1e-12):
            intercepts = shifted.max(axis=0)
        intercepts = np.where(intercepts > 0, intercepts, 1.0)
        return intercepts

    @staticmethod
    def _associate(normalized_F, ref_dirs_norm):
        """Associate solutions with reference directions."""
        norms = np.linalg.norm(normalized_F, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1e-12)
        normalized_vectors = normalized_F / norms
        cosine = normalized_vectors @ ref_dirs_norm.T
        cosine = np.clip(cosine, -1.0, 1.0)
        associations = np.argmax(cosine, axis=1)
        cos_selected = cosine[np.arange(cosine.shape[0]), associations]
        distances = norms.flatten() * np.sqrt(1.0 - np.square(cos_selected))
        return associations, distances

    def _niche_selection(
        self, front, n_remaining, niche_counts, associations, distances, rng
    ):
        """Perform niche-based selection from critical front."""
        selected = []
        pool = front.tolist()
        while len(selected) < n_remaining and pool:
            assoc_front = np.array([associations[idx] for idx in pool])
            unique_refs = np.unique(assoc_front)
            ref_counts = niche_counts[unique_refs]
            min_count = np.min(ref_counts)
            candidate_refs = unique_refs[ref_counts == min_count]
            ref_choice = rng.choice(candidate_refs)

            candidates = [idx for idx in pool if associations[idx] == ref_choice]
            if not candidates:
                niche_counts[ref_choice] = np.inf
                continue
            cand_dist = np.array([distances[idx] for idx in candidates])
            best = candidates[np.argmin(cand_dist)]
            pool.remove(best)
            niche_counts[ref_choice] += 1
            selected.append(best)
        if len(selected) < n_remaining and pool:
            selected.extend(rng.choice(pool, size=n_remaining - len(selected), replace=False))
        return np.asarray(selected, dtype=int)

    @staticmethod
    def _fast_non_dominated_sort(F):
        """Fast non-dominated sorting."""
        n = F.shape[0]
        S = [[] for _ in range(n)]
        domination_count = np.zeros(n, dtype=int)
        ranks = np.zeros(n, dtype=int)
        fronts = [[]]

        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                    S[p].append(q)
                elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                    domination_count[p] += 1
            if domination_count[p] == 0:
                ranks[p] = 0
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        ranks[q] = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        fronts.pop()  # remove last empty front
        return fronts

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

        # NSGA-III survival selection
        st.X, st.F, st.G = self._nsga3_survival(
            st.X, st.F, st.G, X, F, G,
            st.pop_size, st.ref_dirs_norm, st.rng
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

        return build_nsga3_result(self._st, hv_reached)

    @property
    def state(self) -> NSGA3State | None:
        """Access current algorithm state."""
        return self._st
