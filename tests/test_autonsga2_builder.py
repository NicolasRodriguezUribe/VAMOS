import numpy as np

from vamos.algorithm.autonsga2_builder import build_autonsga2


class _DummyProblem:
    def __init__(self, n_var: int = 3):
        self.n_var = n_var
        self.n_obj = 2
        self.xl = np.zeros(n_var)
        self.xu = np.ones(n_var)
        self.encoding = "continuous"

    def evaluate(self, X, out):
        # Simple sphere objectives for compatibility; not used in the builder tests.
        out["F"][:] = np.stack([np.sum(np.square(X), axis=1), np.sum(X, axis=1)], axis=1)


def test_autonsga2_builder_rounds_odd_offspring_size_up():
    cfg = {"population_size": 51}  # offspring_size defaults to pop_size
    algo = build_autonsga2(cfg, _DummyProblem(), seed=0)
    assert algo.cfg["offspring_size"] == 52


def test_autonsga2_builder_accepts_even_offspring_size():
    cfg = {"population_size": 60, "offspring_size": 70}
    algo = build_autonsga2(cfg, _DummyProblem(), seed=1)
    assert algo.cfg["offspring_size"] == 70


def test_autonsga2_builder_supports_extra_continuous_crossovers():
    cfg = {
        "population_size": 40,
        "crossover.type": "pcx",
        "crossover.pcx_sigma_eta": 0.2,
        "crossover.pcx_sigma_zeta": 0.3,
    }
    algo = build_autonsga2(cfg, _DummyProblem(), seed=2)
    method, params = algo.cfg["crossover"]
    assert method == "pcx"
    assert params["sigma_eta"] == 0.2 and params["sigma_zeta"] == 0.3


def test_autonsga2_builder_supports_extra_continuous_mutations():
    cfg = {
        "population_size": 50,
        "mutation.type": "gaussian",
        "mutation.gaussian_sigma": 0.2,
    }
    algo = build_autonsga2(cfg, _DummyProblem(), seed=3)
    method, params = algo.cfg["mutation"]
    assert method == "gaussian"
    assert params["sigma"] == 0.2
