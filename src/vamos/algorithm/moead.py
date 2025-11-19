import numpy as np

from vamos.operators.real import SBXCrossover, PolynomialMutation

from .weight_vectors import load_or_generate_weight_vectors


class MOEAD:
    """
    Simplified MOEA/D implementation that reuses the existing kernel operators
    (SBX crossover + polynomial mutation) and focuses on the decomposition loop.
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
        if pop_size < 2:
            raise ValueError("MOEA/D requires pop_size >= 2.")

        xl = np.asarray(problem.xl, dtype=float)
        xu = np.asarray(problem.xu, dtype=float)
        n_var = problem.n_var
        n_obj = problem.n_obj
        if xl.ndim == 0:
            xl = np.full(n_var, xl, dtype=float)
        if xu.ndim == 0:
            xu = np.full(n_var, xu, dtype=float)

        weight_cfg = self.cfg.get("weight_vectors", {}) or {}
        weight_path = weight_cfg.get("path")
        divisions = weight_cfg.get("divisions")
        weights = load_or_generate_weight_vectors(
            pop_size, n_obj, path=weight_path, divisions=divisions
        )

        neighbor_size = self.cfg.get("neighbor_size", min(20, pop_size))
        neighbor_size = max(2, min(neighbor_size, pop_size))
        neighbors = self._compute_neighbors(weights, neighbor_size)

        aggregation = self.cfg.get("aggregation", ("tchebycheff", {}))
        agg_method, agg_params = aggregation
        aggregator = self._build_aggregator(agg_method, agg_params)

        delta = float(self.cfg.get("delta", 0.9))
        replace_limit = max(1, int(self.cfg.get("replace_limit", 2)))

        cross_method, cross_params = self.cfg["crossover"]
        assert cross_method == "sbx"
        cross_params = dict(cross_params)

        mut_method, mut_params = self.cfg["mutation"]
        assert mut_method == "pm"
        mut_params = dict(mut_params)
        if mut_params.get("prob") == "1/n":
            mut_params["prob"] = 1.0 / n_var

        cross_prob = float(cross_params.get("prob", 0.9))
        cross_eta = float(cross_params.get("eta", 20.0))
        crossover_operator = SBXCrossover(
            prob_crossover=cross_prob,
            eta=cross_eta,
            lower=xl,
            upper=xu,
        )

        mut_prob = float(mut_params.get("prob", 1.0 / max(1, n_var)))
        mut_eta = float(mut_params.get("eta", 20.0))
        mutation_operator = PolynomialMutation(
            prob_mutation=mut_prob,
            eta=mut_eta,
            lower=xl,
            upper=xu,
        )

        X = rng.uniform(xl, xu, size=(pop_size, n_var))
        F = np.empty((pop_size, n_obj))
        problem.evaluate(X, {"F": F})
        n_eval = pop_size

        ideal = F.min(axis=0)
        all_indices = np.arange(pop_size)

        while n_eval < max_eval:
            order = rng.permutation(pop_size)
            remaining = max_eval - n_eval
            batch_size = min(pop_size, remaining)
            active = order[:batch_size]

            parent_pairs = np.empty((batch_size, 2), dtype=int)
            for pos, i in enumerate(active):
                mating_pool = neighbors[i] if rng.random() < delta else all_indices
                if mating_pool.size < 2:
                    mating_pool = all_indices
                parent_pairs[pos] = rng.choice(mating_pool, size=2, replace=False)

            parents_flat = parent_pairs.reshape(-1)
            parents = X[parents_flat].reshape(batch_size, 2, n_var)
            offspring = crossover_operator(parents, rng)
            children = offspring[:, 0, :].copy()
            children = mutation_operator(children, rng)

            F_child = np.empty((batch_size, n_obj))
            problem.evaluate(children, {"F": F_child})
            n_eval += batch_size

            for pos, i in enumerate(active):
                child_vec = children[pos]
                child_f = F_child[pos]
                ideal = np.minimum(ideal, child_f)

                self._update_neighborhood(
                    idx=i,
                    child=child_vec,
                    child_f=child_f,
                    X=X,
                    F=F,
                    weights=weights,
                    neighbors=neighbors,
                    ideal=ideal,
                    replace_limit=replace_limit,
                    aggregator=aggregator,
                    rng=rng,
                )

        return {"X": X, "F": F, "weights": weights}

    @staticmethod
    def _compute_neighbors(weights: np.ndarray, neighbor_size: int) -> np.ndarray:
        dist = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=2)
        order = np.argsort(dist, axis=1)
        return order[:, :neighbor_size]

    @staticmethod
    def _build_aggregator(name: str, params: dict):
        method = name.lower()
        if method == "tchebycheff" or method == "tchebychef":
            return MOEAD._tchebycheff
        if method == "weighted_sum":
            return MOEAD._weighted_sum
        raise ValueError(f"Unsupported aggregation method '{name}'.")

    @staticmethod
    def _tchebycheff(fvals: np.ndarray, weights: np.ndarray, ideal: np.ndarray) -> np.ndarray:
        diff = np.abs(fvals - ideal)
        return np.max(weights * diff, axis=1)

    @staticmethod
    def _weighted_sum(fvals: np.ndarray, weights: np.ndarray, ideal: np.ndarray) -> np.ndarray:
        shifted = fvals - ideal
        return np.sum(weights * shifted, axis=1)

    def _update_neighborhood(
        self,
        idx: int,
        child: np.ndarray,
        child_f: np.ndarray,
        X: np.ndarray,
        F: np.ndarray,
        weights: np.ndarray,
        neighbors: np.ndarray,
        ideal: np.ndarray,
        replace_limit: int,
        aggregator,
        rng: np.random.Generator,
    ):
        neighbor_idx = neighbors[idx]
        if neighbor_idx.size == 0:
            return

        local_weights = weights[neighbor_idx]
        current_vals = aggregator(F[neighbor_idx], local_weights, ideal)

        child_vals = aggregator(child_f, local_weights, ideal)

        improved_mask = child_vals < current_vals
        if not np.any(improved_mask):
            return

        candidates = neighbor_idx[improved_mask]
        if candidates.size > replace_limit:
            replace_idx = rng.choice(candidates.size, size=replace_limit, replace=False)
            candidates = candidates[replace_idx]

        X[candidates] = child
        F[candidates] = child_f
