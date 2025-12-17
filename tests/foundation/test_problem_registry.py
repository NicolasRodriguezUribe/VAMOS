import pytest

from vamos.foundation.problem.registry import make_problem_selection


def test_problem_selection_defaults_and_encoding():
    selection = make_problem_selection("zdt1")
    assert selection.n_var == 30
    assert selection.n_obj == 2
    problem = selection.instantiate()
    assert getattr(problem, "encoding", "continuous") == "continuous"


def test_tsplib_selection_encoding():
    selection = make_problem_selection("kroa100")
    assert selection.n_var == 100
    assert selection.n_obj == 2
    problem = selection.instantiate()
    assert getattr(problem, "encoding", "") == "permutation"


def test_invalid_problem_raises():
    with pytest.raises(KeyError):
        make_problem_selection("does_not_exist")
