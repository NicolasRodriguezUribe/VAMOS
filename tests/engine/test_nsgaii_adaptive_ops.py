import numpy as np

from vamos.engine.algorithm.nsgaii import NSGAII
from vamos.foundation.kernel.numpy_backend import NumPyKernel


class DummyProblem:
    def __init__(self):
        self.n_var = 2
        self.n_obj = 2
        self.xl = np.array([0.0, 0.0])
        self.xu = np.array([1.0, 1.0])
        self.encoding = "continuous"

    def evaluate(self, X, out):
        out["F"] = X.copy()


def test_nsga2_with_adaptive_operator_selector_runs():
    cfg = {
        "pop_size": 10,
        "offspring_size": 10,
        "crossover": ("sbx", {"prob": 0.9, "eta": 15.0}),
        "mutation": ("pm", {"prob": "1/n", "eta": 20.0}),
        "selection": ("tournament", {"pressure": 2}),
        "engine": "numpy",
        "result_mode": "population",
        "adaptive_operator_selection": {
            "enabled": True,
            "method": "ucb",
            "operator_pool": [
                {"crossover": ("sbx", {"prob": 0.9, "eta": 15.0}), "mutation": ("pm", {"prob": "1/n", "eta": 20.0})},
                {"crossover": ("blx_alpha", {"prob": 0.9, "alpha": 0.5}), "mutation": ("gaussian", {"prob": "1/n", "sigma": 0.1})},
            ],
        },
    }
    algo = NSGAII(cfg, NumPyKernel())
    problem = DummyProblem()
    result = algo.run(problem, termination=("max_evaluations", 30), seed=0)
    assert "F" in result and result["F"].shape[0] == cfg["pop_size"]
