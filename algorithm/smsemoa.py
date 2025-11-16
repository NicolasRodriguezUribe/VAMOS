import numpy as np

from .hypervolume import hypervolume_contributions


class SMSEMOA:
    """
    SMS-EMOA (S-Metric Selection Evolutionary Multiobjective Algorithm).
    """

    def __init__(self, config: dict, kernel):
        self.cfg = config
        self.kernel = kernel

    def run(self, problem, termination, seed: int):
        term_type, term_val = termination
        assert term_type == "n_eval", "Only termination=('n_eval', N) is supported."
        max_eval = term_val

        rng = np.random.default_rng(seed)

        pop_size = self.cfg["pop_size"]
        xl, xu = problem.xl, problem.xu
        n_var = problem.n_var

        cross_method, cross_params = self.cfg["crossover"]
        assert cross_method == "sbx"
        cross_params = dict(cross_params)

        mut_method, mut_params = self.cfg["mutation"]
        assert mut_method == "pm"
        mut_params = dict(mut_params)
        if mut_params.get("prob") == "1/n":
            mut_params["prob"] = 1.0 / n_var

        sel_method, sel_params = self.cfg["selection"]
        assert sel_method == "tournament"
        pressure = sel_params.get("pressure", 2)

        ref_cfg = self.cfg.get("reference_point", {}) or {}

        # Initialization
        X = rng.uniform(xl, xu, size=(pop_size, n_var))
        F = np.empty((pop_size, problem.n_obj))
        problem.evaluate(X, {"F": F})
        n_eval = pop_size

        ref_point, ref_offset, ref_adaptive = self._initialize_reference_point(F, ref_cfg)

        while n_eval < max_eval:
            ranks, crowd = self.kernel.nsga2_ranking(F)
            parents_idx = self.kernel.tournament_selection(
                ranks, crowd, pressure, rng, n_parents=2
            )
            parents = X[parents_idx]
            offspring = self.kernel.sbx_crossover(parents, cross_params, rng, xl, xu)
            child = offspring[:1].copy()
            self.kernel.polynomial_mutation(child, mut_params, rng, xl, xu)

            F_child = np.empty((1, problem.n_obj))
            problem.evaluate(child, {"F": F_child})
            n_eval += 1

            if ref_adaptive:
                ref_point = np.maximum(ref_point, F_child[0] + ref_offset)

            X, F = self._survival(
                X,
                F,
                child[0],
                F_child[0],
                ref_point,
                pop_size,
            )

        return {"X": X, "F": F, "reference_point": ref_point}

    @staticmethod
    def _initialize_reference_point(F: np.ndarray, ref_cfg: dict):
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

    def _survival(
        self,
        X: np.ndarray,
        F: np.ndarray,
        child_x: np.ndarray,
        child_f: np.ndarray,
        ref_point: np.ndarray,
        pop_size: int,
    ):
        X_comb = np.vstack([X, child_x])
        F_comb = np.vstack([F, child_f])

        ranks, _ = self.kernel.nsga2_ranking(F_comb)
        worst_rank = ranks.max()
        worst_idx = np.flatnonzero(ranks == worst_rank)

        if worst_idx.size == 1:
            remove_idx = worst_idx[0]
        else:
            contribs = hypervolume_contributions(F_comb[worst_idx], ref_point)
            remove_idx = worst_idx[np.argmin(contribs)]

        keep = np.delete(np.arange(F_comb.shape[0]), remove_idx)
        return X_comb[keep][:pop_size], F_comb[keep][:pop_size]
