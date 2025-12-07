import numpy as np

from vamos.operators.binary import (
    random_binary_population,
    one_point_crossover,
    two_point_crossover,
    uniform_crossover,
    bit_flip_mutation,
)
from vamos.operators.integer import (
    random_integer_population,
    uniform_integer_crossover,
    arithmetic_integer_crossover,
    random_reset_mutation,
    creep_mutation,
)
from vamos.operators.real import SBXCrossover, PolynomialMutation, VariationWorkspace
from .hypervolume import hypervolume_contributions


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
        encoding = getattr(problem, "encoding", "continuous")
        bounds_dtype = int if encoding == "integer" else float
        xl = np.asarray(problem.xl, dtype=bounds_dtype)
        xu = np.asarray(problem.xu, dtype=bounds_dtype)
        n_var = problem.n_var
        if xl.ndim == 0:
            xl = np.full(n_var, xl, dtype=bounds_dtype)
        if xu.ndim == 0:
            xu = np.full(n_var, xu, dtype=bounds_dtype)

        cross_method, cross_params = self.cfg["crossover"]
        cross_params = dict(cross_params)

        mut_method, mut_params = self.cfg["mutation"]
        mut_params = dict(mut_params)
        if mut_params.get("prob") == "1/n":
            mut_params["prob"] = 1.0 / n_var

        sel_method, sel_params = self.cfg["selection"]
        assert sel_method == "tournament"
        pressure = sel_params.get("pressure", 2)

        ref_cfg = self.cfg.get("reference_point", {}) or {}

        # Build variation per encoding.
        if encoding == "binary":
            if cross_method not in _BINARY_CROSSOVER:
                raise ValueError(f"Unsupported SMSEMOA crossover '{cross_method}' for binary encoding.")
            if mut_method not in _BINARY_MUTATION:
                raise ValueError(f"Unsupported SMSEMOA mutation '{mut_method}' for binary encoding.")
            cross_fn = _BINARY_CROSSOVER[cross_method]
            cross_prob = float(cross_params.get("prob", 0.9))
            mut_fn = _BINARY_MUTATION[mut_method]
            mut_prob = _resolve_prob_expression(mut_params.get("prob"), n_var, 1.0 / max(1, n_var))
            crossover = lambda parents: cross_fn(parents, cross_prob, rng)
            mutation = lambda X_child: (mut_fn(X_child, mut_prob, rng) or X_child)
            X = random_binary_population(pop_size, n_var, rng)
        elif encoding == "integer":
            if cross_method not in _INT_CROSSOVER:
                raise ValueError(f"Unsupported SMSEMOA crossover '{cross_method}' for integer encoding.")
            if mut_method not in _INT_MUTATION:
                raise ValueError(f"Unsupported SMSEMOA mutation '{mut_method}' for integer encoding.")
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
            X = random_integer_population(pop_size, n_var, xl.astype(int), xu.astype(int), rng)
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
            mut_prob = _resolve_prob_expression(mut_params.get("prob"), n_var, 1.0 / max(1, n_var))
            mut_eta = float(mut_params.get("eta", 20.0))
            pm = PolynomialMutation(
                prob_mutation=mut_prob,
                eta=mut_eta,
                lower=xl,
                upper=xu,
                workspace=workspace,
            )
            crossover = lambda parents: sbx(parents, rng)
            mutation = lambda X_child: pm(X_child, rng)
            X = rng.uniform(xl, xu, size=(pop_size, n_var))
        else:
            raise ValueError(f"SMSEMOA does not support encoding '{encoding}'.")

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
            if parents.ndim == 2:
                parents = parents.reshape(1, 2, n_var)
            offspring = crossover(parents)
            child_vec = offspring.reshape(-1, n_var)[0:1]  # first child as (1, n_var)
            child = mutation(child_vec)

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
