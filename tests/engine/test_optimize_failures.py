import pytest

from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.experiment.optimize import OptimizeConfig, optimize_config
from vamos.foundation.exceptions import InvalidAlgorithmError
from vamos.foundation.problem.zdt1 import ZDT1Problem


def test_unknown_algorithm_name_errors():
    cfg = OptimizeConfig(
        problem=ZDT1Problem(n_var=4),
        algorithm="does_not_exist",
        algorithm_config=NSGAIIConfig.default(pop_size=4, n_var=4),
        termination=("n_eval", 4),
        seed=0,
    )
    with pytest.raises(InvalidAlgorithmError, match="Unknown algorithm"):
        optimize_config(cfg)


def test_unknown_problem_selection_errors():
    cfg = OptimizeConfig(
        problem="nonexistent_problem",  # type: ignore[arg-type]
        algorithm="nsgaii",
        algorithm_config=NSGAIIConfig.default(pop_size=4, n_var=4),
        termination=("n_eval", 4),
        seed=0,
    )
    with pytest.raises(Exception):
        optimize_config(cfg)
