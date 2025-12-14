import numpy as np

from vamos.foundation.eval.backends import SerialEvalBackend, MultiprocessingEvalBackend


class DummyProblem:
    def __init__(self):
        self.n_var = 2
        self.n_obj = 1
        self.n_constr = 0
        self.xl = -5.0
        self.xu = 5.0
        self.encoding = "real"

    def evaluate(self, X, out):
        out["F"] = np.sum(X * X, axis=1, keepdims=True)


def test_serial_eval_backend_matches_direct():
    prob = DummyProblem()
    X = np.array([[1.0, 2.0], [0.5, -0.5]])
    backend = SerialEvalBackend()

    res = backend.evaluate(X, prob)

    expected = np.sum(X * X, axis=1, keepdims=True)
    np.testing.assert_allclose(res.F, expected)
    assert res.G is None


def test_multiprocessing_eval_backend_matches_serial():
    prob = DummyProblem()
    X = np.array([[1.0, 2.0], [0.5, -0.5], [3.0, 0.0]])
    serial = SerialEvalBackend().evaluate(X, prob)
    mp = MultiprocessingEvalBackend(n_workers=2).evaluate(X, prob)

    np.testing.assert_allclose(mp.F, serial.F)
    assert mp.G is None
