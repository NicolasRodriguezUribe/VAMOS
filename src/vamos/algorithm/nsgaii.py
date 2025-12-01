# algorithm/nsgaii.py
import numpy as np

from vamos.algorithm.archive import CrowdingDistanceArchive, CrowdingArchive, _single_front_crowding
from vamos.algorithm.population import evaluate_population, initialize_population, resolve_bounds
from vamos.algorithm.termination import HVTracker
from vamos.algorithm.variation import (
    VariationPipeline,
    prepare_mutation_params,
)
from vamos.operators.real import VariationWorkspace


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
    if selection_method == "random":
        parent_indices = rng.integers(0, ranks.size, size=parent_count)
    else:
        parent_indices = kernel.tournament_selection(
            ranks, crowding, pressure, rng, n_parents=parent_count
        )
    if parent_indices.size != parent_count:
        raise ValueError("Selection operator returned an unexpected number of parents.")
    return parent_indices.reshape(parent_count // group_size, group_size)



class NSGAII:
    """
    Vectorized/SOA-style NSGA-II evolutionary core.
    Individuals are represented as array rows (X, F) without per-object instances.
    """

    def __init__(self, config: dict, kernel):
        self.cfg = config
        self.kernel = kernel

    def run(self, problem, termination, seed: int):
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

        rng = np.random.default_rng(seed)
        pop_size = int(self.cfg["pop_size"])
        offspring_size = int(self.cfg.get("offspring_size") or pop_size)
        if offspring_size <= 0:
            raise ValueError("offspring size must be positive.")
        if offspring_size % 2 != 0:
            raise ValueError("offspring size must be even.")
        encoding = getattr(problem, "encoding", "continuous")
        n_var = problem.n_var
        xl, xu = resolve_bounds(problem, encoding)

        initializer_cfg = self.cfg.get("initializer")
        X = initialize_population(pop_size, n_var, xl, xu, encoding, rng, problem, initializer=initializer_cfg)
        F = evaluate_population(problem, X)
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
        hv_points = lambda: archive_F if archive_F is not None and archive_F.size else F
        hv_reached = hv_tracker.reached(hv_points()) if hv_tracker.enabled else False

        result_mode = self.cfg.get("result_mode", "population")
        crowd_archive = None
        if result_mode == "external_archive" and archive_size:
            crowd_archive = CrowdingArchive(
                archive_size,
                dominance_fn=lambda a, b: 1 if np.all(a <= b) and np.any(a < b) else (-1 if np.all(b <= a) and np.any(b < a) else 0),
                crowding_fn=_single_front_crowding,
            )

        while n_eval < max_eval and not hv_reached:
            ranks, crowding = self.kernel.nsga2_ranking(F)
            parents_per_group = variation.parents_per_group
            children_per_group = variation.children_per_group
            parent_count = int(np.ceil(offspring_size / children_per_group) * parents_per_group)
            mating_pairs = _build_mating_pool(
                self.kernel, ranks, crowding, pressure, rng, parent_count, parents_per_group, sel_method
            )
            parent_idx = mating_pairs.reshape(-1)
            X_parents = variation.gather_parents(X, parent_idx)

            X_off = variation.produce_offspring(X_parents, rng)
            if X_off.shape[0] > offspring_size:
                X_off = X_off[:offspring_size]

            F_off = evaluate_population(problem, X_off)
            n_eval += X_off.shape[0]

            X, F = self.kernel.nsga2_survival(X, F, X_off, F_off, pop_size)
            if crowd_archive is not None:
                for xx, ff in zip(X, F):
                    crowd_archive.add(xx, ff)
            if archive_size:
                if archive_via_kernel:
                    archive_X, archive_F = self.kernel.update_archive(
                        archive_X, archive_F, X, F, archive_size
                    )
                elif archive_manager is not None:
                    archive_X, archive_F = archive_manager.update(X, F)
            if hv_tracker.enabled and hv_tracker.reached(hv_points()):
                hv_reached = True
                break

        result = {"X": X, "F": F, "evaluations": n_eval, "hv_reached": hv_reached}
        if crowd_archive is not None:
            arch_X, arch_F = crowd_archive.get_solutions()
            result["archive"] = {"X": arch_X, "F": arch_F}
        elif archive_size:
            if archive_manager is not None:
                final_X, final_F = archive_manager.contents()
                result["archive"] = {"X": final_X, "F": final_F}
            elif archive_via_kernel:
                result["archive"] = {"X": archive_X, "F": archive_F}
        return result

    def _resolve_archive_size(self) -> int | None:
        archive_cfg = self.cfg.get("archive") or self.cfg.get("external_archive")
        if not archive_cfg:
            return None
        size = int(archive_cfg.get("size", 0))
        return size if size > 0 else None
