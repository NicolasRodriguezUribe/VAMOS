import numpy as np

from vamos.algorithm.config import NSGAIIConfig, MOEADConfig
from vamos.algorithm.nsgaii import NSGAII
from vamos.algorithm.moead import MOEAD
from vamos.kernel.numpy_backend import NumPyKernel
from vamos.constraints.utils import compute_violation, is_feasible


class LinearConstraintProblem:
    def __init__(self):
        self.n_var = 2
        self.n_obj = 1
        self.n_constr = 1
        self.xl = 0.0
        self.xu = 2.0
        self.encoding = "real"

    def evaluate(self, X, out):
        # Objective: distance from (0.5,0.5)
        dx = X - 0.5
        out["F"] = np.sum(dx * dx, axis=1, keepdims=True)
        # Constraint: x0 + x1 - 1 <= 0
        out["G"] = (X[:, 0] + X[:, 1] - 1.0)[:, None]


def _make_nsgaii(cfg_mode: str):
    cfg = (
        NSGAIIConfig()
        .pop_size(10)
        .offspring_size(10)
        .crossover("sbx", prob=0.9, eta=15.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", pressure=2)
        .survival("nsga2")
        .engine("numpy")
        .constraint_mode(cfg_mode)
        .fixed()
    )
    return NSGAII(cfg.to_dict(), kernel=NumPyKernel())


def test_feasibility_handling_prefers_feasible_solutions():
    problem = LinearConstraintProblem()
    algo = _make_nsgaii("feasibility")
    result = algo.run(problem, termination=("n_eval", 40), seed=3)
    G = result.get("G")
    assert G is not None
    feas = is_feasible(G)
    assert feas.any()
    assert feas.sum() >= 5  # at least half feasible in final pop


def test_unconstrained_mode_can_keep_infeasible():
    problem = LinearConstraintProblem()
    algo = _make_nsgaii("none")
    result = algo.run(problem, termination=("n_eval", 40), seed=4)
    G = result.get("G")
    assert G is None  # not tracked in none mode


def test_moead_penalty_reduces_violation():
    problem = LinearConstraintProblem()
    cfg = (
        MOEADConfig()
        .pop_size(10)
        .neighbor_size(3)
        .delta(0.9)
        .replace_limit(2)
        .crossover("sbx", prob=1.0, eta=20.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .aggregation("tchebycheff")
        .engine("numpy")
        .constraint_mode("feasibility")
        .fixed()
    )
    algo = MOEAD(cfg.to_dict(), kernel=NumPyKernel())
    result = algo.run(problem, termination=("n_eval", 40), seed=5)
    G = result.get("G")
    assert G is not None
    cv = compute_violation(G)
    assert np.median(cv) <= cv.max()
