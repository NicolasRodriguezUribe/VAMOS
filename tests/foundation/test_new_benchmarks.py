import numpy as np
from vamos.foundation.problem.lz import LZ09F1Problem, LZ09F6Problem
from vamos.foundation.problem.cec import CEC2009CF1Problem, CEC2009UF1Problem, CEC2009UF4Problem, CEC2009UF8Problem
from vamos.foundation.problem.constrained_many import CDTLZProblem, DCDTLZProblem, MWProblem
from vamos.foundation.problem.dtlz import DTLZ5Problem, DTLZ6Problem
from vamos.foundation.problem.lsmop import LSMOP1, LSMOP9
from vamos.foundation.problem.real_world.engineering import WeldedBeamDesignProblem
from vamos.foundation.problem.zdt5 import ZDT5Problem


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


def test_cec2009_extended_uf_smoke():
    problem = CEC2009UF4Problem()
    X = _sample(problem, n=3)
    out = {"F": np.empty((X.shape[0], problem.n_obj))}
    problem.evaluate(X, out)
    assert out["F"].shape == (3, 2)
    assert np.isfinite(out["F"]).all()

    problem3 = CEC2009UF8Problem()
    X3 = _sample(problem3, n=2)
    out3 = {"F": np.empty((X3.shape[0], problem3.n_obj))}
    problem3.evaluate(X3, out3)
    assert out3["F"].shape == (2, 3)
    assert np.isfinite(out3["F"]).all()


def test_welded_beam_design_constraints():
    problem = WeldedBeamDesignProblem()
    X = _sample(problem, n=2)
    out = {"F": np.empty((X.shape[0], problem.n_obj))}
    problem.evaluate(X, out)
    assert out["F"].shape == (2, problem.n_obj)
    assert "G" in out
    assert out["G"].shape[0] == X.shape[0]


def test_dtlz56_and_zdt5_smoke():
    problem = DTLZ5Problem(n_var=12, n_obj=3)
    X = _sample(problem, n=3)
    out = {"F": np.empty((X.shape[0], problem.n_obj))}
    problem.evaluate(X, out)
    assert out["F"].shape == (3, problem.n_obj)
    assert np.isfinite(out["F"]).all()

    problem2 = DTLZ6Problem(n_var=12, n_obj=3)
    X2 = _sample(problem2, n=2)
    out2 = {"F": np.empty((X2.shape[0], problem2.n_obj))}
    problem2.evaluate(X2, out2)
    assert out2["F"].shape == (2, problem2.n_obj)
    assert np.isfinite(out2["F"]).all()

    zdt5 = ZDT5Problem(n_var=80)
    rng = np.random.default_rng(123)
    Xb = rng.integers(0, 2, size=(2, zdt5.n_var)).astype(float)
    outb = {"F": np.empty((Xb.shape[0], zdt5.n_obj))}
    zdt5.evaluate(Xb, outb)
    assert outb["F"].shape == (2, zdt5.n_obj)
    assert np.isfinite(outb["F"]).all()

    X_ones = np.ones((1, zdt5.n_var), dtype=float)
    out_ones = {"F": np.empty((1, zdt5.n_obj))}
    zdt5.evaluate(X_ones, out_ones)
    assert out_ones["F"][0, 0] == 31.0
    assert np.isclose(out_ones["F"][0, 1], 10.0 / 31.0)


def test_lsmop_smoke():
    problem = LSMOP1(n_var=300, n_obj=2)
    X = _sample(problem, n=3)
    out = {"F": np.empty((X.shape[0], problem.n_obj))}
    problem.evaluate(X, out)
    assert out["F"].shape == (3, 2)
    assert np.isfinite(out["F"]).all()

    problem9 = LSMOP9(n_var=300, n_obj=2)
    X9 = _sample(problem9, n=2)
    out9 = {"F": np.empty((X9.shape[0], problem9.n_obj))}
    problem9.evaluate(X9, out9)
    assert out9["F"].shape == (2, 2)
    assert np.isfinite(out9["F"]).all()


def test_constrained_many_wrappers_smoke():
    cdtlz = CDTLZProblem("c1dtlz1", n_var=12, n_obj=2)
    X1 = _sample(cdtlz, n=3)
    out1 = {"F": np.empty((X1.shape[0], cdtlz.n_obj))}
    cdtlz.evaluate(X1, out1)
    assert out1["F"].shape == (3, 2)
    assert "G" in out1
    assert out1["G"].shape[0] == X1.shape[0]

    dcdtlz = DCDTLZProblem("dc2dtlz3", n_var=12, n_obj=2)
    X2 = _sample(dcdtlz, n=2)
    out2 = {"F": np.empty((X2.shape[0], dcdtlz.n_obj))}
    dcdtlz.evaluate(X2, out2)
    assert out2["F"].shape == (2, 2)
    assert "G" in out2
    assert out2["G"].shape == (2, 2)

    mw = MWProblem("mw1", n_var=15, n_obj=2)
    X2 = _sample(mw, n=2)
    out3 = {"F": np.empty((X2.shape[0], mw.n_obj))}
    mw.evaluate(X2, out3)
    assert out3["F"].shape == (2, 2)
    assert "G" in out3
    assert out3["G"].shape[0] == X2.shape[0]
