# algorithm/nsgaii.py
import numpy as np

from vamos.operators.permutation import (
    order_crossover,
    random_permutation_population,
    swap_mutation,
)


def _resolve_prob_expression(value, n_var: int, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, str) and value.endswith("/n"):
        numerator = value[:-2]
        num = float(numerator) if numerator else 1.0
        return min(1.0, max(num, 0.0) / n_var)
    return float(value)


def _initialize_population(
    pop_size: int,
    n_var: int,
    xl,
    xu,
    encoding: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if pop_size <= 0:
        raise ValueError("pop_size must be positive.")
    if encoding == "permutation":
        return random_permutation_population(pop_size, n_var, rng)
    return rng.uniform(xl, xu, size=(pop_size, n_var))


def _evaluate_population(problem, X: np.ndarray) -> np.ndarray:
    F = np.empty((X.shape[0], problem.n_obj))
    problem.evaluate(X, {"F": F})
    return F


def _prepare_mutation_params(mut_params: dict, encoding: str, n_var: int) -> dict:
    params = dict(mut_params)
    if "prob" in params:
        params["prob"] = _resolve_prob_expression(params["prob"], n_var, params["prob"])
    else:
        if encoding == "permutation":
            params["prob"] = min(1.0, 2.0 / max(1, n_var))
        else:
            params["prob"] = 1.0 / max(1, n_var)
    return params


def _build_mating_pool(
    kernel,
    ranks: np.ndarray,
    crowding: np.ndarray,
    pressure: int,
    rng: np.random.Generator,
    pop_size: int,
) -> np.ndarray:
    if pop_size % 2 != 0:
        raise ValueError("NSGA-II requires an even population size.")
    parent_count = pop_size
    parent_indices = kernel.tournament_selection(
        ranks, crowding, pressure, rng, n_parents=parent_count
    )
    if parent_indices.size != parent_count:
        raise ValueError("Selection operator returned an unexpected number of parents.")
    return parent_indices.reshape(pop_size // 2, 2)


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
        if term_type != "n_eval":
            raise ValueError("Only termination=('n_eval', N) is supported by NSGA-II.")
        max_eval = int(term_val)

        rng = np.random.default_rng(seed)
        pop_size = int(self.cfg["pop_size"])
        encoding = getattr(problem, "encoding", "continuous")
        n_var = problem.n_var
        xl, xu = problem.xl, problem.xu

        X = _initialize_population(pop_size, n_var, xl, xu, encoding, rng)
        F = _evaluate_population(problem, X)
        n_eval = X.shape[0]

        sel_method, sel_params = self.cfg["selection"]
        if sel_method != "tournament":
            raise ValueError(f"Unsupported selection method '{sel_method}'.")
        pressure = int(sel_params.get("pressure", 2))

        cross_method, cross_params = self.cfg["crossover"]
        cross_params = dict(cross_params)

        mut_method, mut_params = self.cfg["mutation"]
        mut_params = _prepare_mutation_params(mut_params, encoding, n_var)

        self._validate_operator_support(encoding, cross_method, mut_method)

        while n_eval < max_eval:
            ranks, crowding = self.kernel.nsga2_ranking(F)
            mating_pairs = _build_mating_pool(
                self.kernel, ranks, crowding, pressure, rng, pop_size
            )
            parent_idx = mating_pairs.reshape(-1)
            X_parents = X[parent_idx]

            X_off = self._produce_offspring(
                X_parents,
                encoding,
                cross_method,
                cross_params,
                mut_method,
                mut_params,
                rng,
                xl,
                xu,
            )

            F_off = _evaluate_population(problem, X_off)
            n_eval += X_off.shape[0]

            X, F = self.kernel.nsga2_survival(X, F, X_off, F_off, pop_size)

        return {"X": X, "F": F}

    @staticmethod
    def _validate_operator_support(encoding: str, crossover: str, mutation: str) -> None:
        if encoding == "permutation":
            if crossover not in {"ox", "order"}:
                raise ValueError(f"Unsupported crossover '{crossover}' for permutation encoding.")
            if mutation != "swap":
                raise ValueError(f"Unsupported mutation '{mutation}' for permutation encoding.")
        else:
            if crossover != "sbx":
                raise ValueError(f"Unsupported crossover '{crossover}' for continuous encoding.")
            if mutation != "pm":
                raise ValueError(f"Unsupported mutation '{mutation}' for continuous encoding.")

    def _produce_offspring(
        self,
        parents: np.ndarray,
        encoding: str,
        crossover: str,
        cross_params: dict,
        mutation: str,
        mut_params: dict,
        rng: np.random.Generator,
        xl,
        xu,
    ) -> np.ndarray:
        if encoding == "permutation":
            cross_prob = float(cross_params.get("prob", 1.0))
            offspring = order_crossover(parents, cross_prob, rng)
            swap_mutation(offspring, float(mut_params.get("prob", 0.0)), rng)
            return offspring

        offspring = self.kernel.sbx_crossover(parents, cross_params, rng, xl, xu)
        self.kernel.polynomial_mutation(offspring, mut_params, rng, xl, xu)
        return offspring
