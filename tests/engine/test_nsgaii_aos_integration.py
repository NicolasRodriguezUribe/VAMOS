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


def _base_cfg():
    return {
        "pop_size": 10,
        "offspring_size": 10,
        "crossover": ("sbx", {"prob": 0.9, "eta": 15.0}),
        "mutation": ("pm", {"prob": "1/n", "eta": 20.0}),
        "selection": ("tournament", {"pressure": 2}),
        "engine": "numpy",
        "result_mode": "population",
    }


def test_nsgaii_aos_disabled_is_noop():
    cfg = _base_cfg()
    algo = NSGAII(cfg, NumPyKernel())
    problem = DummyProblem()
    result = algo.run(problem, termination=("n_eval", 20), seed=0)
    assert "aos" not in result
    st = algo._st
    assert st is not None
    assert st.aos_controller is None


def test_nsgaii_aos_enabled_produces_trace_rows():
    cfg = _base_cfg()
    cfg["adaptive_operator_selection"] = {
        "enabled": True,
        "method": "epsilon_greedy",
        "reward_scope": "survival",
        "epsilon": 0.2,
        "min_usage": 1,
        "rng_seed": 0,
        "operator_pool": [
            {
                "crossover": ("sbx", {"prob": 0.9, "eta": 15.0}),
                "mutation": ("pm", {"prob": "1/n", "eta": 20.0}),
            },
            {
                "crossover": ("blx_alpha", {"prob": 0.9, "alpha": 0.5}),
                "mutation": ("gaussian", {"prob": "1/n", "sigma": 0.1}),
            },
        ],
    }
    algo = NSGAII(cfg, NumPyKernel())
    problem = DummyProblem()
    result = algo.run(problem, termination=("n_eval", 30), seed=0)
    aos = result.get("aos")
    assert aos is not None
    rows = aos.get("trace_rows") or []
    assert len(rows) == 2
    for row in rows:
        for key in (
            "step",
            "mating_id",
            "op_id",
            "op_name",
            "reward",
            "reward_survival",
            "reward_nd_insertions",
            "reward_hv_delta",
            "batch_size",
        ):
            assert key in row
        assert 0.0 <= row["reward"] <= 1.0
        assert 0.0 <= row["reward_survival"] <= 1.0
        assert 0.0 <= row["reward_nd_insertions"] <= 1.0
        assert 0.0 <= row["reward_hv_delta"] <= 1.0
        assert row["batch_size"] > 0
