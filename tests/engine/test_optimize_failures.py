import pytest

from vamos.experiment.optimize import OptimizeConfig, optimize
from vamos.foundation.problem.zdt1 import ZDT1Problem


def test_unknown_algorithm_name_errors():
    cfg = OptimizeConfig(
        problem=ZDT1Problem(n_var=4),
        algorithm="does_not_exist",
        algorithm_config={},
        termination=("n_eval", 4),
        seed=0,
    )
    with pytest.raises(ValueError, match="Unknown algorithm"):
        optimize(cfg)


def test_unknown_problem_selection_errors():
    cfg = OptimizeConfig(
        problem="nonexistent_problem",  # type: ignore[arg-type]
        algorithm="nsgaii",
        algorithm_config={},
        termination=("n_eval", 4),
        seed=0,
    )
    with pytest.raises(Exception):
        optimize(cfg)
