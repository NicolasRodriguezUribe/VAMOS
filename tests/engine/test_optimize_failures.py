import pytest

from vamos import optimize
from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.foundation.exceptions import InvalidAlgorithmError
from vamos.foundation.problem.zdt1 import ZDT1Problem


def test_unknown_algorithm_name_errors():
    with pytest.raises(InvalidAlgorithmError, match="Unknown algorithm"):
        optimize(ZDT1Problem(n_var=4), algorithm="does_not_exist", max_evaluations=4, pop_size=4, seed=0)


def test_unknown_problem_selection_errors():
    with pytest.raises(Exception):
        optimize("nonexistent_problem", algorithm="nsgaii", max_evaluations=4, pop_size=4, seed=0)


def test_invalid_max_evaluations_errors():
    with pytest.raises(ValueError, match="max_evaluations must be a positive integer"):
        optimize(ZDT1Problem(n_var=4), algorithm="nsgaii", max_evaluations=0, pop_size=4, seed=0)


def test_invalid_pop_size_errors():
    with pytest.raises(ValueError, match="pop_size must be a positive integer"):
        optimize(ZDT1Problem(n_var=4), algorithm="nsgaii", max_evaluations=4, pop_size=-1, seed=0)


def test_invalid_algorithm_config_pop_size_errors():
    cfg = (
        NSGAIIConfig.builder()
        .pop_size(-4)
        .crossover("sbx", prob=1.0, eta=20.0)
        .mutation("pm", prob=0.1, eta=20.0)
        .selection("tournament")
        .build()
    )
    with pytest.raises(ValueError, match=r"algorithm_config\.pop_size"):
        optimize(
            ZDT1Problem(n_var=4),
            algorithm="nsgaii",
            algorithm_config=cfg,
            termination=("max_evaluations", 4),
            seed=0,
        )


def test_invalid_eval_strategy_errors():
    with pytest.raises(ValueError, match="eval_strategy must be one of"):
        optimize(
            ZDT1Problem(n_var=4),
            algorithm="nsgaii",
            max_evaluations=4,
            pop_size=4,
            seed=0,
            eval_strategy="parallel",
        )
