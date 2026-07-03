import numpy as np
import pytest

from vamos.foundation.problem.tsp import TSPProblem
from vamos.foundation.problem.zdt1 import ZDT1Problem
from vamos.foundation.problem.zdt2 import ZDT2Problem


@pytest.mark.parametrize(
    ("problem_cls", "name"),
    [
        (ZDT1Problem, "ZDT1"),
        (ZDT2Problem, "ZDT2"),
    ],
)
def test_zdt1_zdt2_reject_one_decision_variable(problem_cls, name: str) -> None:
    with pytest.raises(ValueError, match=f"{name} requires at least two decision variables"):
        problem_cls(n_var=1)


def test_tsp_rejects_out_of_range_city_labels() -> None:
    problem = TSPProblem(n_cities=4)
    out = {"F": np.zeros((1, 2))}

    with pytest.raises(ValueError, match="city labels"):
        problem.evaluate(np.array([[0, 1, 2, 7]]), out)


def test_tsp_rejects_duplicate_city_labels() -> None:
    problem = TSPProblem(n_cities=4)
    out = {"F": np.zeros((1, 2))}

    with pytest.raises(ValueError, match="valid permutations"):
        problem.evaluate(np.array([[0, 1, 1, 3]]), out)


def test_tsp_rejects_fractional_city_labels() -> None:
    problem = TSPProblem(n_cities=4)
    out = {"F": np.zeros((1, 2))}

    with pytest.raises(ValueError, match="integer city labels"):
        problem.evaluate(np.array([[0.0, 1.0, 2.5, 3.0]]), out)
