# algorithm/nsgaii.py
import numpy as np

from vamos.algorithm.archive import CrowdingDistanceArchive, HypervolumeArchive, _single_front_crowding
from vamos.algorithm.population import initialize_population, resolve_bounds
from vamos.algorithm.termination import HVTracker
from vamos.algorithm.variation import (
    VariationPipeline,
    prepare_mutation_params,
)
from vamos.operators.real import VariationWorkspace
from vamos.constraints.utils import compute_violation, is_feasible
from vamos.eval.backends import SerialEvalBackend


def _build_mating_pool(
    kernel,
    ranks: np.ndarray,
    crowding: np.ndarray,
    pressure: int,
    rng: np.random.Generator,
    parent_count: int,
    group_size: int = 2,
    selection_method: str = "tournament",
) -> np.ndarray:
    if parent_count <= 0:
        raise ValueError("parent_count must be positive.")
    if parent_count % group_size != 0:
        raise ValueError("parent_count must be divisible by group_size.")
    assert ranks.shape == crowding.shape, "ranks and crowding must align"
    if selection_method == "random":
        parent_indices = rng.integers(0, ranks.size, size=parent_count)
    else:
        parent_indices = kernel.tournament_selection(
            ranks, crowding, pressure, rng, n_parents=parent_count
        )
    if parent_indices.size != parent_count:
        raise ValueError("Selection operator returned an unexpected number of parents.")
    return parent_indices.reshape(parent_count // group_size, group_size)


def _feasible_nsga2_survival(kernel, X, F, G, X_off, F_off, G_off, pop_size):
    """
    Feasibility rule:
      - Feasible dominate infeasible.
      - Among feasible: standard NSGA-II rank/crowding.
      - Among infeasible: lower sum violation wins.
    """
    X_comb = np.vstack([X, X_off])
    F_comb = np.vstack([F, F_off])
    G_comb = np.vstack([G, G_off]) if G is not None and G_off is not None else None
    if G_comb is None:
        X_sel, F_sel = kernel.nsga2_survival(X, F, X_off, F_off, pop_size)
        return X_sel, F_sel, None

    feas = is_feasible(G_comb)
    cv = compute_violation(G_comb)
    selected = []

    if feas.any():
        feas_idx = np.nonzero(feas)[0]
        ranks, crowd = kernel.nsga2_ranking(F_comb[feas_idx])
        # Order feasible by rank then crowding
        order = np.lexsort((-crowd, ranks))
        feas_ordered = feas_idx[order]
        selected.extend(feas_ordered.tolist())

    if len(selected) < pop_size:
        infeas_idx = np.nonzero(~feas)[0]
        if infeas_idx.size:
            order_infeas = infeas_idx[np.argsort(cv[infeas_idx])]
            selected.extend(order_infeas.tolist())

    selected = selected[:pop_size]
    return X_comb[selected], F_comb[selected], G_comb[selected]


class NSGAII:
    """
    Vectorized/SOA-style NSGA-II evolutionary core.
    Individuals are represented as array rows (X, F) without per-object instances.
    """

    def __init__(self, config: dict, kernel):
        self.cfg = config
        self.kernel = kernel

    def run(self, problem, termination, seed: int, eval_backend=None):
        term_type, term_val = termination
        hv_config = None
        if term_type == "n_eval":
            max_eval = int(term_val)
        elif term_type == "hv":
            hv_config = dict(term_val)
            max_eval = int(hv_config.get("max_evaluations", 0))
            if max_eval <= 0:
                raise ValueError("HV-based termination requires a positive max_evaluations value.")
        else:
            raise ValueError("Unsupported termination criterion for NSGA-II.")

        if eval_backend is None:
            eval_backend = SerialEvalBackend()
        rng = np.random.default_rng(seed)
        pop_size = int(self.cfg["pop_size"])
        offspring_size = int(self.cfg.get("offspring_size") or pop_size)
        if offspring_size <= 0:
            raise ValueError("offspring size must be positive.")
        encoding = getattr(problem, "encoding", "continuous")
        n_var = problem.n_var
        xl, xu = resolve_bounds(problem, encoding)

        initializer_cfg = self.cfg.get("initializer")
        X = initialize_population(pop_size, n_var, xl, xu, encoding, rng, problem, initializer=initializer_cfg)
        constraint_mode = self.cfg.get("constraint_mode", "none")
        eval_result = eval_backend.evaluate(X, problem)
        F = eval_result.F
        G = eval_result.G if constraint_mode != "none" else None
        assert X.shape[0] == F.shape[0], "Population and objectives must align"
        n_eval = X.shape[0]
        hv_tracker = HVTracker(hv_config, self.kernel)
        archive_size = self._resolve_archive_size()
        archive_X = archive_F = None
        archive_manager = None
        archive_via_kernel = False
        if archive_size:
            if hasattr(self.kernel, "update_archive"):
                archive_via_kernel = True
                archive_X, archive_F = self.kernel.update_archive(
                    None, None, X, F, archive_size
                )
            else:
                archive_manager = CrowdingDistanceArchive(
                    archive_size, n_var, problem.n_obj, X.dtype
                )
                archive_X, archive_F = archive_manager.update(X, F)

        sel_method, sel_params = self.cfg["selection"]
        if sel_method != "tournament":
            if sel_method != "random":
                raise ValueError(f"Unsupported selection method '{sel_method}'.")
        pressure = int(sel_params.get("pressure", 2)) if sel_method == "tournament" else 2

        cross_method, cross_params = self.cfg["crossover"]
        cross_method = cross_method.lower()
        cross_params = dict(cross_params)

        mut_method, mut_params = self.cfg["mutation"]
        mut_method = mut_method.lower()
        mut_factor = self.cfg.get("mutation_prob_factor")
        mut_params = prepare_mutation_params(mut_params, encoding, n_var, prob_factor=mut_factor)

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
        result_mode = self.cfg.get("result_mode", "population")
        result_archive = None
        if result_mode == "external_archive" and archive_size:
            archive_type = self.cfg.get("archive_type", "hypervolume")
            if archive_type == "crowding":
                result_archive = CrowdingDistanceArchive(
                    archive_size, n_var, problem.n_obj, X.dtype
                )
            else:
                result_archive = HypervolumeArchive(
                    archive_size, n_var, problem.n_obj, X.dtype
                )

        # Initialize ask/tell state
        self._state = {
            "X": X,
            "F": F,
            "G": G,
            "rng": rng,
            "variation": variation,
            "offspring_size": offspring_size,
            "sel_method": sel_method,
            "pressure": pressure,
            "constraint_mode": constraint_mode,
            "pop_size": pop_size,
            "archive_size": archive_size,
            "archive_X": archive_X,
            "archive_F": archive_F,
            "archive_manager": archive_manager,
            "archive_via_kernel": archive_via_kernel,
            "result_archive": result_archive,
            "hv_tracker": hv_tracker,
        }
        self._state["hv_points_fn"] = lambda: (
            self._state["archive_F"]
            if self._state.get("archive_F") is not None and self._state["archive_F"].size
            else self._state["F"]
        )
        hv_reached = hv_tracker.enabled and hv_tracker.reached(self._state["hv_points_fn"]())

        while n_eval < max_eval and not hv_reached:
            X_off = self.ask()
            eval_off = eval_backend.evaluate(X_off, problem)
            hv_reached = self.tell(eval_off, pop_size)
            n_eval += X_off.shape[0]
            X = self._state["X"]
            F = self._state["F"]
            G = self._state["G"]
            archive_X = self._state.get("archive_X")
            archive_F = self._state.get("archive_F")
            hv_points_fn = self._state["hv_points_fn"]
            if hv_tracker.enabled and hv_tracker.reached(hv_points_fn()):
                hv_reached = True
                break

        result = {"X": self._state["X"], "F": self._state["F"], "evaluations": n_eval, "hv_reached": hv_reached}
        if G is not None:
            result["G"] = self._state["G"]
        if result_archive is not None:
            arch_X, arch_F = result_archive.contents()
            result["archive"] = {"X": arch_X, "F": arch_F}
        elif archive_size:
            archive_manager = self._state.get("archive_manager")
            archive_via_kernel = self._state.get("archive_via_kernel", False)
            archive_X = self._state.get("archive_X")
            archive_F = self._state.get("archive_F")
            if archive_manager is not None:
                final_X, final_F = archive_manager.contents()
                result["archive"] = {"X": final_X, "F": final_F}
            elif archive_via_kernel:
                result["archive"] = {"X": archive_X, "F": archive_F}
        return result

    def _selection_metrics(self, F: np.ndarray, G: np.ndarray | None, constraint_mode: str):
        ranks, crowding = self.kernel.nsga2_ranking(F)
        if G is not None and constraint_mode != "none":
            cv = compute_violation(G)
            feas = is_feasible(G)
            if feas.any():
                feas_idx = np.nonzero(feas)[0]
                feas_ranks, feas_crowd = self.kernel.nsga2_ranking(F[feas_idx])
                ranks = np.full(F.shape[0], feas_ranks.max(initial=0) + 1, dtype=int)
                crowding = np.zeros(F.shape[0], dtype=float)
                ranks[feas_idx] = feas_ranks
                crowding[feas_idx] = feas_crowd
                crowding[~feas] = -cv[~feas]
            else:
                ranks = np.zeros(F.shape[0], dtype=int)
                crowding = -cv
        return ranks, crowding

    def ask(self) -> np.ndarray:
        """Generate offspring from the current state (minimal ask/tell support)."""
        if not hasattr(self, "_state"):
            raise RuntimeError("ask() called before initialization.")
        st = self._state
        ranks, crowding = self._selection_metrics(st["F"], st["G"], st["constraint_mode"])
        parents_per_group = st["variation"].parents_per_group
        children_per_group = st["variation"].children_per_group
        parent_count = int(np.ceil(st["offspring_size"] / children_per_group) * parents_per_group)
        mating_pairs = _build_mating_pool(
            self.kernel,
            ranks,
            crowding,
            st["pressure"],
            st["rng"],
            parent_count,
            parents_per_group,
            st["sel_method"],
        )
        parent_idx = mating_pairs.reshape(-1)
        X_parents = st["variation"].gather_parents(st["X"], parent_idx)
        X_off = st["variation"].produce_offspring(X_parents, st["rng"])
        if X_off.shape[0] > st["offspring_size"]:
            X_off = X_off[: st["offspring_size"]]
        st["pending_offspring"] = X_off
        return X_off

    def tell(self, eval_result, pop_size: int) -> bool:
        """Consume evaluated offspring and update state. Returns hv_reached flag."""
        if not hasattr(self, "_state"):
            raise RuntimeError("tell() called before initialization.")
        st = self._state
        X_off = st.pop("pending_offspring", None)
        if X_off is None:
            raise ValueError("tell() called without a pending ask().")
        F_off = eval_result.F
        G_off = eval_result.G if st["constraint_mode"] != "none" else None

        if st["G"] is None or st["constraint_mode"] == "none":
            st["X"], st["F"] = self.kernel.nsga2_survival(st["X"], st["F"], X_off, F_off, pop_size)
            st["G"] = None
        else:
            st["X"], st["F"], st["G"] = _feasible_nsga2_survival(
                self.kernel, st["X"], st["F"], st["G"], X_off, F_off, G_off, pop_size
            )
        if st["result_archive"] is not None:
            st["result_archive"].update(st["X"], st["F"])
        archive_size = st["archive_size"]
        if archive_size:
            if st["archive_via_kernel"]:
                st["archive_X"], st["archive_F"] = self.kernel.update_archive(
                    st["archive_X"], st["archive_F"], st["X"], st["F"], archive_size
                )
            elif st["archive_manager"] is not None:
                st["archive_X"], st["archive_F"] = st["archive_manager"].update(st["X"], st["F"])
        hv_tracker = st["hv_tracker"]
        hv_points = st["hv_points_fn"]
        hv_reached = hv_tracker.enabled and hv_tracker.reached(hv_points())
        return hv_reached

    def _resolve_archive_size(self) -> int | None:
        archive_cfg = self.cfg.get("archive") or self.cfg.get("external_archive")
        if not archive_cfg:
            return None
        size = int(archive_cfg.get("size", 0))
        return size if size > 0 else None
