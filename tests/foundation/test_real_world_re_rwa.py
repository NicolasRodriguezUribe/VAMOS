import numpy as np
import pytest

from vamos.foundation.problem.registry import make_problem_selection


def _sample(problem, n: int = 4, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xl = problem.xl
    xu = problem.xu
    if np.isscalar(xl) or np.isscalar(xu):
        return rng.uniform(float(xl), float(xu), size=(n, problem.n_var))
    return rng.uniform(np.asarray(xl, dtype=float), np.asarray(xu, dtype=float), size=(n, problem.n_var))


@pytest.mark.parametrize(
    "key",
    [
        # Zapotecas-Martínez et al. (2023) RWA suite
        "rwa1",
        "rwa2",
        "rwa3",
        "rwa4",
        "rwa5",
        "rwa6",
        "rwa7",
        "rwa8",
        "rwa9",
        "rwa10",
        # Tanabe & Ishibuchi (2020) RE suite
        "re21",
        "re22",
        "re23",
        "re24",
        "re25",
        "re31",
        "re32",
        "re33",
        "re34",
        "re35",
        "re36",
        "re37",
        "re41",
        "re42",
        "re61",
        "re91",
    ],
)
def test_re_and_rwa_shapes_and_finiteness(key: str):
    selection = make_problem_selection(key)
    problem = selection.instantiate()
    X = _sample(problem, n=5)
    out = {"F": np.empty((X.shape[0], problem.n_obj))}
    problem.evaluate(X, out)
    assert out["F"].shape == (X.shape[0], problem.n_obj)
    assert np.isfinite(out["F"]).all()


@pytest.mark.parametrize("key", ["re22", "re23", "re25", "re35"])
def test_re_mixed_problems_expose_mixed_spec(key: str):
    problem = make_problem_selection(key).instantiate()
    assert getattr(problem, "encoding", None) == "mixed"
    assert hasattr(problem, "mixed_spec")
