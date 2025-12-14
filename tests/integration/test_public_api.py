from vamos import (
    NSGAIIConfig,
    available_algorithms,
    available_problem_names,
    friedman_test,
    optimize,
    plot_pareto_front_2d,
    weighted_sum_scores,
)
from vamos.problems import ZDT1


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

