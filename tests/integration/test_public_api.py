from vamos.api import available_problem_names, optimize
from vamos.engine.algorithms_registry import available_algorithms
from vamos.engine.api import NSGAIIConfig
from vamos.foundation.problems_registry import ZDT1
from vamos.ux.api import friedman_test, plot_pareto_front_2d, weighted_sum_scores


def test_public_api_symbols_exist():
    assert callable(optimize)
    assert NSGAIIConfig is not None
    assert ZDT1 is not None
    assert callable(plot_pareto_front_2d)
    assert callable(weighted_sum_scores)
    assert callable(friedman_test)


def test_available_helpers_return_values():
    assert "nsgaii" in available_algorithms()
    assert "zdt1" in available_problem_names()
