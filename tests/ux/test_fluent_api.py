import pytest
from vamos.api import OptimizationResult
from vamos.experiment.builder import study


@pytest.mark.smoke
def test_fluent_api_basic():
    """Verify simplest usage: study(problem).run()"""
    res = study("zdt1", n_var=10).run()
    assert isinstance(res, OptimizationResult)
    assert res.F is not None
    assert len(res.F) == 100  # Default pop size


@pytest.mark.smoke
def test_fluent_api_advanced():
    """Verify chained configuration."""
    res = study("zdt1", n_var=10).using("nsgaii", pop_size=20).engine("numpy").evaluations(1000).seed(42).run()
    assert isinstance(res, OptimizationResult)
    assert len(res.F) == 20
    assert res.X.shape[1] == 10


@pytest.mark.smoke
def test_fluent_api_engine_switch():
    """Verify switching engine works via builder."""
    # Using 'numba' if available, otherwise just check it accepts string
    res = (
        study("zdt1", n_var=10)
        .using("nsgaii")
        .engine("numba")  # Valid engine
        .evaluations(100)
        .run()
    )
    assert isinstance(res, OptimizationResult)
