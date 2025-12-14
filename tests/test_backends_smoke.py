import pytest

from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.problem.registry import make_problem_selection
from vamos.foundation.core.runner import run_single


@pytest.mark.numba
def test_numba_backend_smoke(monkeypatch):
    pytest.importorskip("numba")
    selection = make_problem_selection("zdt1", n_var=6)
    cfg = ExperimentConfig(population_size=6, offspring_population_size=6, max_evaluations=20, seed=2)
    result = run_single("numba", "nsgaii", selection, cfg)
    assert result["F"].shape[0] > 0
    assert result["engine"] == "numba"


@pytest.mark.moocore
def test_moocore_backend_smoke(monkeypatch):
    pytest.importorskip("moocore")
    selection = make_problem_selection("zdt1", n_var=6)
    cfg = ExperimentConfig(population_size=6, offspring_population_size=6, max_evaluations=20, seed=3)
    result = run_single("moocore", "nsgaii", selection, cfg)
    assert result["F"].shape[0] > 0
    assert result["engine"] == "moocore"
