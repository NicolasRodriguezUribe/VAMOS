# algorithm/nsgaii.py
import numpy as np


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
        assert term_type == "n_eval", "Only termination=('n_eval', N) is supported in this demo"
        max_eval = term_val

        rng = np.random.default_rng(seed)
        pop_size = self.cfg["pop_size"]
        xl, xu = problem.xl, problem.xu
        n_var = problem.n_var
        n_eval = 0

        # Initialization
        X = rng.uniform(xl, xu, size=(pop_size, n_var))
        F = np.empty((pop_size, problem.n_obj))
        problem.evaluate(X, {"F": F})
        n_eval += pop_size

        # Unpack the configuration
        sel_method, sel_params = self.cfg["selection"]
        assert sel_method == "tournament"
        pressure = sel_params.get("pressure", 2)

        cross_method, cross_params = self.cfg["crossover"]
        assert cross_method == "sbx"
        cross_params = dict(cross_params)

        mut_method, mut_params = self.cfg["mutation"]
        assert mut_method == "pm"
        mut_params = dict(mut_params)
        # Resolve prob="1/n" now that n_var is known
        if mut_params.get("prob") == "1/n":
            mut_params["prob"] = 1.0 / n_var

        # Evolution loop
        while n_eval < max_eval:
            # Ranking / crowding for the current fronts
            ranks, crowd = self.kernel.nsga2_ranking(F)

            # Parent selection (pairs) -> 2 * n_pairs indices
            n_pairs = pop_size // 2
            parents_idx = self.kernel.tournament_selection(
                ranks, crowd, pressure, rng, n_parents=2 * n_pairs
            )
            X_parents = X[parents_idx]

            # SBX crossover
            X_off = self.kernel.sbx_crossover(X_parents, cross_params, rng, xl, xu)

            # Polynomial mutation (in-place)
            self.kernel.polynomial_mutation(X_off, mut_params, rng, xl, xu)

            # Evaluate offspring
            F_off = np.empty((X_off.shape[0], problem.n_obj))
            problem.evaluate(X_off, {"F": F_off})
            n_eval += X_off.shape[0]

            # NSGA-II survival (elitism)
            X, F = self.kernel.nsga2_survival(X, F, X_off, F_off, pop_size)

        return {"X": X, "F": F}
