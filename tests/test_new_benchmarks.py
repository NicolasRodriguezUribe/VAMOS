import numpy as np
import pytest

from vamos.problem.lz import LZ09F1Problem, LZ09F6Problem
from vamos.problem.cec import CEC2009UF1Problem, CEC2009CF1Problem
from vamos.problem.real_world.engineering import WeldedBeamDesignProblem


def _sample(problem, n: int = 4) -> np.ndarray:
    rng = np.random.default_rng(123)
    xl = problem.xl
    xu = problem.xu
    if np.isscalar(xl) or np.isscalar(xu):
        return rng.uniform(float(xl), float(xu), size=(n, problem.n_var))
    return rng.uniform(np.asarray(xl, dtype=float), np.asarray(xu, dtype=float), size=(n, problem.n_var))


def test_lz09_shapes_and_finiteness():
    problem = LZ09F6Problem()  # tri-objective variant
    X = _sample(problem, n=3)
    out = {"F": np.empty((X.shape[0], problem.n_obj))}
    problem.evaluate(X, out)
    assert out["F"].shape == (3, problem.n_obj)
    assert np.isfinite(out["F"]).all()

    problem2 = LZ09F1Problem()
    X2 = _sample(problem2, n=2)
    out2 = {"F": np.empty((X2.shape[0], problem2.n_obj))}
    problem2.evaluate(X2, out2)
    assert out2["F"].shape == (2, problem2.n_obj)
    assert np.isfinite(out2["F"]).all()


def test_cec2009_uf_and_cf_smoke():
    problem = CEC2009UF1Problem()
    X = _sample(problem, n=3)
    out = {"F": np.empty((X.shape[0], problem.n_obj))}
    problem.evaluate(X, out)
    assert out["F"].shape == (3, problem.n_obj)
    assert np.isfinite(out["F"]).all()

    cf_problem = CEC2009CF1Problem()
    Xc = _sample(cf_problem, n=2)
    out_c = {"F": np.empty((Xc.shape[0], cf_problem.n_obj))}
    cf_problem.evaluate(Xc, out_c)
    assert out_c["F"].shape == (2, cf_problem.n_obj)
    assert "G" in out_c
    assert out_c["G"].shape[0] == Xc.shape[0]


def test_welded_beam_design_constraints():
    problem = WeldedBeamDesignProblem()
    X = _sample(problem, n=2)
    out = {"F": np.empty((X.shape[0], problem.n_obj))}
    problem.evaluate(X, out)
    assert out["F"].shape == (2, problem.n_obj)
    assert "G" in out
    assert out["G"].shape[0] == X.shape[0]
