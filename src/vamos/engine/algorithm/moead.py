import numpy as np

from vamos.engine.operators.binary import (
    random_binary_population,
    one_point_crossover,
    two_point_crossover,
    uniform_crossover,
    bit_flip_mutation,
)
from vamos.engine.operators.integer import (
    random_integer_population,
    uniform_integer_crossover,
    arithmetic_integer_crossover,
    random_reset_mutation,
    creep_mutation,
)
from vamos.engine.operators.real import SBXCrossover, PolynomialMutation, VariationWorkspace
from vamos.engine.algorithm.population import evaluate_population_with_constraints
from vamos.foundation.constraints.utils import compute_violation, is_feasible

from .weight_vectors import load_or_generate_weight_vectors


def _resolve_prob_expression(value, n_var: int, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, str) and value.endswith("/n"):
        numerator = value[:-2]
        num = float(numerator) if numerator else 1.0
        return min(1.0, max(num, 0.0) / n_var)
    return float(value)


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
    Simplified MOEA/D implementation that reuses the existing kernel operators
    for real-coded problems and dedicated binary/integer operators for discrete encodings.
    Focuses on the decomposition loop.
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
        constraint_mode = self.cfg.get("constraint_mode", "feasibility")

        encoding = getattr(problem, "encoding", "continuous")
        bounds_dtype = int if encoding == "integer" else float
        xl = np.asarray(problem.xl, dtype=bounds_dtype)
        xu = np.asarray(problem.xu, dtype=bounds_dtype)
        n_var = problem.n_var
        n_obj = problem.n_obj
        if xl.ndim == 0:
            xl = np.full(n_var, xl, dtype=bounds_dtype)
        if xu.ndim == 0:
            xu = np.full(n_var, xu, dtype=bounds_dtype)

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
        cross_params = dict(cross_params)

        mut_method, mut_params = self.cfg["mutation"]
        mut_params = dict(mut_params)
        if mut_params.get("prob") == "1/n":
            mut_params["prob"] = 1.0 / n_var

        # Build variation operators per encoding (all return mutated arrays).
        if encoding == "binary":
            if cross_method not in _BINARY_CROSSOVER:
                raise ValueError(f"Unsupported MOEA/D crossover '{cross_method}' for binary encoding.")
            if mut_method not in _BINARY_MUTATION:
                raise ValueError(f"Unsupported MOEA/D mutation '{mut_method}' for binary encoding.")
            cross_fn = _BINARY_CROSSOVER[cross_method]
            cross_prob = float(cross_params.get("prob", 0.9))
            mut_fn = _BINARY_MUTATION[mut_method]
            mut_prob = _resolve_prob_expression(mut_params.get("prob"), n_var, 1.0 / max(1, n_var))
            crossover = lambda parents: cross_fn(parents, cross_prob, rng)
            mutation = lambda X_child: (mut_fn(X_child, mut_prob, rng) or X_child)
        elif encoding == "integer":
            if cross_method not in _INT_CROSSOVER:
                raise ValueError(f"Unsupported MOEA/D crossover '{cross_method}' for integer encoding.")
            if mut_method not in _INT_MUTATION:
                raise ValueError(f"Unsupported MOEA/D mutation '{mut_method}' for integer encoding.")
            cross_fn = _INT_CROSSOVER[cross_method]
            cross_prob = float(cross_params.get("prob", 0.9))
            mut_fn = _INT_MUTATION[mut_method]
            mut_prob = _resolve_prob_expression(mut_params.get("prob"), n_var, 1.0 / max(1, n_var))
            step = int(mut_params.get("step", 1))
            crossover = lambda parents: cross_fn(parents, cross_prob, rng)
            mutation = (
                (lambda X_child: (mut_fn(X_child, mut_prob, step, xl, xu, rng) or X_child))
                if mut_fn is creep_mutation
                else (lambda X_child: (mut_fn(X_child, mut_prob, xl, xu, rng) or X_child))
            )
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
            mut_prob = _resolve_prob_expression(mut_params.get("prob"), n_var, 1.0 / max(1, n_var))
            mut_eta = float(mut_params.get("eta", 20.0))
            mutation_operator = PolynomialMutation(
                prob_mutation=mut_prob,
                eta=mut_eta,
                lower=xl,
                upper=xu,
                workspace=workspace,
            )
            crossover = lambda parents: crossover_operator(parents, rng)
            mutation = lambda X_child: mutation_operator(X_child, rng)
        else:
            raise ValueError(f"MOEA/D does not support encoding '{encoding}'.")

        if encoding == "binary":
            X = random_binary_population(pop_size, n_var, rng)
        elif encoding == "integer":
            X = random_integer_population(pop_size, n_var, xl.astype(int), xu.astype(int), rng)
        else:
            X = rng.uniform(xl, xu, size=(pop_size, n_var))
        F, G = evaluate_population_with_constraints(problem, X)
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
            offspring = crossover(parents)
            children = offspring[:, 0, :].copy()
            children = mutation(children)

            F_child, G_child = evaluate_population_with_constraints(problem, children)
            n_eval += batch_size

            for pos, i in enumerate(active):
                child_vec = children[pos]
                child_f = F_child[pos]
                ideal = np.minimum(ideal, child_f)

                self._update_neighborhood(
                    idx=i,
                    child=child_vec,
                    child_f=child_f,
                    child_g=G_child[pos] if G_child is not None else None,
                    X=X,
                    F=F,
                    G=G,
                    weights=weights,
                    neighbors=neighbors,
                    ideal=ideal,
                    replace_limit=replace_limit,
                    aggregator=aggregator,
                    rng=rng,
                    constraint_mode=constraint_mode,
                    cv_penalty=compute_violation(G_child)[pos] if G_child is not None else 0.0,
                )

        result = {"X": X, "F": F, "weights": weights}
        if G is not None:
            result["G"] = G
        return result

    @staticmethod
    def _compute_neighbors(weights: np.ndarray, neighbor_size: int) -> np.ndarray:
        dist = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=2)
        order = np.argsort(dist, axis=1)
        return order[:, :neighbor_size]

    @staticmethod
    def _build_aggregator(name: str, params: dict):
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
        diff = np.abs(fvals - ideal)
        return np.max(weights * diff, axis=1)

    @staticmethod
    def _weighted_sum(fvals: np.ndarray, weights: np.ndarray, ideal: np.ndarray) -> np.ndarray:
        shifted = fvals - ideal
        return np.sum(weights * shifted, axis=1)

    @staticmethod
    def _pbi(fvals: np.ndarray, weights: np.ndarray, ideal: np.ndarray, theta: float) -> np.ndarray:
        """
        Penalty boundary intersection (PBI).
        """
        diff = fvals - ideal
        norm_w = np.linalg.norm(weights, axis=1, keepdims=True)
        norm_w = np.where(norm_w > 0, norm_w, 1.0)
        w_unit = weights / norm_w
        d1 = np.abs(np.sum(diff * w_unit, axis=1))
        proj = (d1[:, None]) * w_unit
        d2 = np.linalg.norm(diff - proj, axis=1)
        return d1 + theta * d2

    @staticmethod
    def _modified_tchebycheff(fvals: np.ndarray, weights: np.ndarray, ideal: np.ndarray, rho: float) -> np.ndarray:
        """
        Modified Tchebycheff: max component plus a weighted L1 term (scaled by rho).
        """
        diff = np.abs(fvals - ideal)
        weighted = weights * diff
        return np.max(weighted, axis=1) + rho * np.sum(weighted, axis=1)

    def _update_neighborhood(
        self,
        idx: int,
        child: np.ndarray,
        child_f: np.ndarray,
        child_g: np.ndarray | None,
        X: np.ndarray,
        F: np.ndarray,
        G: np.ndarray | None,
        weights: np.ndarray,
        neighbors: np.ndarray,
        ideal: np.ndarray,
        replace_limit: int,
        aggregator,
        rng: np.random.Generator,
        *,
        constraint_mode: str = "feasibility",
        cv_penalty: float = 0.0,
    ):
        if constraint_mode == "none" or G is None or child_g is None:
            constraint_mode = "none"
        neighbor_idx = neighbors[idx]
        if neighbor_idx.size == 0:
            return

        local_weights = weights[neighbor_idx]
        current_vals = aggregator(F[neighbor_idx], local_weights, ideal)

        child_vals = aggregator(child_f, local_weights, ideal)
        if constraint_mode != "none":
            child_cv = cv_penalty
            current_cv = compute_violation(G[neighbor_idx]) if G is not None else np.zeros_like(current_vals)
            # Prefer feasible over infeasible; add CV as penalty
            feas_child = child_cv <= 0.0
            feas_curr = current_cv <= 0.0
            # mask where child is strictly better by feasibility rule
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

        if candidates.size > replace_limit:
            replace_idx = rng.choice(candidates.size, size=replace_limit, replace=False)
            candidates = candidates[replace_idx]

        X[candidates] = child
        F[candidates] = child_f
        if G is not None and child_g is not None:
            G[candidates] = child_g
