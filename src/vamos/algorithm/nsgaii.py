# algorithm/nsgaii.py
import numpy as np

from vamos.algorithm.archive import CrowdingDistanceArchive, HypervolumeArchive, _single_front_crowding
from vamos.algorithm.population import initialize_population, resolve_bounds
from vamos.algorithm.termination import HVTracker
from vamos.algorithm.variation import (
    VariationPipeline,
    prepare_mutation_params,
)
from vamos.algorithm.nsgaii_helpers import (
    build_mating_pool,
    feasible_nsga2_survival,
    match_ids,
    operator_success_stats,
    generation_contributions,
)
from vamos.operators.real import VariationWorkspace
from vamos.constraints.utils import compute_violation, is_feasible
from vamos.eval.backends import SerialEvalBackend
from vamos.visualization.live_viz import LiveVisualization, NoOpLiveVisualization
from vamos.hyperheuristics.operator_selector import make_operator_selector, compute_reward
from vamos.hyperheuristics.indicator import IndicatorEvaluator
from vamos.analytics.genealogy import GenealogyTracker, get_lineage

# Backward-compat aliases for modules still importing the old private helpers
_build_mating_pool = build_mating_pool
_feasible_nsga2_survival = feasible_nsga2_survival
_match_ids = match_ids
_operator_success_stats = operator_success_stats
_generation_contributions = generation_contributions


class NSGAII:
    """
    Vectorized/SOA-style NSGA-II evolutionary core.
    Individuals are represented as array rows (X, F) without per-object instances.
    """

    def __init__(self, config: dict, kernel):
        self.cfg = config
        self.kernel = kernel

    def run(self, problem, termination, seed: int, eval_backend=None, live_viz: LiveVisualization | None = None):
        (
            live_cb,
            eval_backend,
            max_eval,
            n_eval,
            hv_tracker,
            hv_reached,
            track_genealogy,
            genealogy_tracker,
        ) = self._initialize_run(problem, termination, seed, eval_backend, live_viz)
        result_archive = self._state.get("result_archive")
        archive_size = self._state.get("archive_size")
        archive_manager = self._state.get("archive_manager")
        archive_via_kernel = self._state.get("archive_via_kernel", False)
        archive_X = self._state.get("archive_X")
        archive_F = self._state.get("archive_F")
        constraint_mode = self._state["constraint_mode"]
        sel_method = self._state["sel_method"]
        pressure = self._state["pressure"]
        pop_size = self._state["pop_size"]
        X = self._state["X"]
        F = self._state["F"]
        G = self._state["G"]
        generation = 0
        live_cb.on_generation(generation, F=self._state["F"])

        while n_eval < max_eval and not hv_reached:
            self._state["generation"] = generation
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
            generation += 1
            self._state["generation"] = generation
            try:
                ranks, _ = self.kernel.nsga2_ranking(F)
                nd_mask = ranks == ranks.min(initial=0)
                live_cb.on_generation(generation, F=F[nd_mask])
            except Exception:
                live_cb.on_generation(generation, F=F)

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
        live_cb.on_end(final_F=self._state["F"])
        if track_genealogy and genealogy_tracker is not None:
            try:
                ranks, _ = self.kernel.nsga2_ranking(self._state["F"])
                nd_mask = ranks == ranks.min(initial=0)
                final_ids = self._state.get("ids")
                final_front_ids = final_ids[nd_mask] if final_ids is not None else []
                genealogy_tracker.mark_final_front(list(final_front_ids))
                result["genealogy"] = {
                    "operator_stats": operator_success_stats(genealogy_tracker, list(final_front_ids)),
                    "generation_contributions": generation_contributions(genealogy_tracker, list(final_front_ids)),
                }
            except Exception:
                pass
        return result

    def _initialize_run(self, problem, termination, seed, eval_backend, live_viz):
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
        live_cb = live_viz or NoOpLiveVisualization()
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
        constraint_mode = self.cfg.get("constraint_mode", "feasibility")
        eval_result = eval_backend.evaluate(X, problem)
        F = eval_result.F
        G = eval_result.G if constraint_mode != "none" else None
        assert X.shape[0] == F.shape[0], "Population and objectives must align"
        n_eval = X.shape[0]
        live_cb.on_start(problem=problem, algorithm=self, config=self.cfg)
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

        track_genealogy = bool(self.cfg.get("track_genealogy", False))
        genealogy_tracker = GenealogyTracker() if track_genealogy else None
        ids = np.arange(pop_size, dtype=int) if track_genealogy else None
        if genealogy_tracker is not None:
            for i in range(pop_size):
                genealogy_tracker.new_individual(
                    generation=0,
                    parents=[],
                    operator_name=None,
                    algorithm_name="nsgaii",
                    fitness=F[i] if F is not None and i < F.shape[0] else None,
                )

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
        operator_pool, op_selector, indicator_eval = self._build_operator_pool(
            encoding,
            cross_method,
            cross_params,
            mut_method,
            mut_params,
            n_var,
            xl,
            xu,
            variation_workspace,
            problem,
            mut_factor,
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

        self._state = {
            "X": X,
            "F": F,
            "G": G,
            "rng": rng,
            "variation": operator_pool[0],
            "operator_pool": operator_pool,
            "op_selector": op_selector,
            "indicator_eval": indicator_eval,
            "last_operator_idx": 0,
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
            "track_genealogy": track_genealogy,
            "genealogy_tracker": genealogy_tracker,
            "ids": ids,
            "generation": 0,
            "variation_workspace": variation_workspace,
            "result_mode": result_mode,
        }
        self._state["hv_points_fn"] = lambda: (
            self._state["archive_F"]
            if self._state.get("archive_F") is not None and self._state["archive_F"].size
            else self._state["F"]
        )
        hv_reached = hv_tracker.enabled and hv_tracker.reached(self._state["hv_points_fn"]())
        return live_cb, eval_backend, max_eval, n_eval, hv_tracker, hv_reached, track_genealogy, genealogy_tracker

    def _build_operator_pool(
        self,
        encoding,
        cross_method,
        cross_params,
        mut_method,
        mut_params,
        n_var,
        xl,
        xu,
        variation_workspace,
        problem,
        mut_factor,
    ):
        operator_pool = []
        op_configs = self.cfg.get("adaptive_operators", {}).get("operator_pool")
        if op_configs:
            for entry in op_configs:
                c_method, c_params = entry.get("crossover", (cross_method, cross_params))
                m_method, m_params = entry.get("mutation", (mut_method, mut_params))
                m_params = prepare_mutation_params(m_params, encoding, n_var, prob_factor=mut_factor)
                operator_pool.append(
                    VariationPipeline(
                        encoding=encoding,
                        cross_method=c_method,
                        cross_params=c_params,
                        mut_method=m_method,
                        mut_params=m_params,
                        xl=xl,
                        xu=xu,
                        workspace=variation_workspace,
                        repair_cfg=self.cfg.get("repair"),
                        problem=problem,
                    )
                )
        if not operator_pool:
            operator_pool.append(
                VariationPipeline(
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
            )
        selector_cfg = self.cfg.get("adaptive_operators", {})
        adaptive_enabled = bool(selector_cfg.get("enabled", False)) and len(operator_pool) > 1
        op_selector = None
        indicator_eval = None
        if adaptive_enabled:
            method = selector_cfg.get("method", "epsilon_greedy")
            op_selector = make_operator_selector(
                method,
                len(operator_pool),
                epsilon=selector_cfg.get("epsilon", 0.1),
                c=selector_cfg.get("c", 1.0),
            )
            indicator = selector_cfg.get("indicator", "hv")
            indicator_mode = selector_cfg.get("mode", "maximize")
            indicator_eval = IndicatorEvaluator(indicator, reference_point=None, mode=indicator_mode)
        return operator_pool, op_selector, indicator_eval

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
        if st.get("op_selector") is not None:
            idx = st["op_selector"].select_operator()
            st["variation"] = st["operator_pool"][idx]
            st["last_operator_idx"] = idx
        ranks, crowding = self._selection_metrics(st["F"], st["G"], st["constraint_mode"])
        parents_per_group = st["variation"].parents_per_group
        children_per_group = st["variation"].children_per_group
        parent_count = int(np.ceil(st["offspring_size"] / children_per_group) * parents_per_group)
        mating_pairs = build_mating_pool(
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
        if st.get("track_genealogy") and st.get("genealogy_tracker") is not None:
            operator_name = f"{st['variation'].cross_method}+{st['variation'].mut_method}"
            group_size = st["variation"].parents_per_group
            children_per_group = st["variation"].children_per_group
            parent_groups = parent_idx.reshape(-1, group_size)
            child_ids = []
            gen = st.get("generation", 0) + 1
            for parents in parent_groups:
                parent_ids = st["ids"][parents] if st.get("ids") is not None else []
                for _ in range(children_per_group):
                    child_ids.append(
                        st["genealogy_tracker"].new_individual(
                            generation=gen,
                            parents=list(parent_ids),
                            operator_name=operator_name,
                            algorithm_name="nsgaii",
                        )
                    )
            st["pending_offspring_ids"] = np.asarray(child_ids[: X_off.shape[0]], dtype=int)
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

        hv_before = None
        if st.get("indicator_eval") is not None:
            try:
                hv_before = st["indicator_eval"].compute(st["hv_points_fn"]())
            except Exception:
                hv_before = None

        combined_X = np.vstack([st["X"], X_off])
        combined_F = np.vstack([st["F"], F_off])
        combined_G = None
        if st["G"] is not None and G_off is not None and st["constraint_mode"] != "none":
            combined_G = np.vstack([st["G"], G_off])
        combined_ids = None
        if st.get("track_genealogy"):
            current_ids = st.get("ids")
            if current_ids is None:
                current_ids = np.array([], dtype=int)
            pending_ids = st.get("pending_offspring_ids")
            if pending_ids is None:
                pending_ids = np.array([], dtype=int)
            combined_ids = np.concatenate([current_ids, pending_ids])

        if combined_G is None or st["constraint_mode"] == "none":
            new_X, new_F = self.kernel.nsga2_survival(st["X"], st["F"], X_off, F_off, pop_size)
            new_G = None
        else:
            new_X, new_F, new_G = feasible_nsga2_survival(
                self.kernel, st["X"], st["F"], st["G"], X_off, F_off, G_off, pop_size
            )

        if combined_ids is not None:
            st["ids"] = match_ids(new_X, combined_X, combined_ids)

        st["X"], st["F"], st["G"] = new_X, new_F, new_G
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
        if st.get("op_selector") is not None and hv_before is not None:
            try:
                hv_after = st["indicator_eval"].compute(hv_points())
                reward = compute_reward(hv_before, hv_after, st["indicator_eval"].mode)
                st["op_selector"].update(st.get("last_operator_idx", 0), reward)
            except Exception:
                pass
        st.pop("pending_offspring_ids", None)
        return hv_reached

    def _resolve_archive_size(self) -> int | None:
        archive_cfg = self.cfg.get("archive") or self.cfg.get("external_archive")
        if not archive_cfg:
            return None
        size = int(archive_cfg.get("size", 0))
        return size if size > 0 else None
