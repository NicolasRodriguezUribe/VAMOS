import pytest

from vamos.foundation.problem.registry import make_problem_selection


def test_problem_selection_defaults_and_encoding():
    selection = make_problem_selection("zdt1")
    assert selection.n_var == 30
    assert selection.n_obj == 2
    problem = selection.instantiate()
    assert problem.encoding == "real"


def test_problem_selection_zdt5_binary_encoding():
    selection = make_problem_selection("zdt5")
    assert selection.n_var == 80
    assert selection.n_obj == 2
    problem = selection.instantiate()
    assert getattr(problem, "encoding", "") == "binary"


@pytest.mark.parametrize("name", ["dtlz5", "dtlz6"])
def test_problem_selection_dtlz56_defaults(name: str):
    selection = make_problem_selection(name)
    assert selection.n_var == 12
    assert selection.n_obj == 3


@pytest.mark.parametrize("name", ["lsmop1", "lsmop9"])
def test_problem_selection_lsmop_defaults(name: str):
    selection = make_problem_selection(name)
    assert selection.n_var == 300
    assert selection.n_obj == 3


@pytest.mark.parametrize("name", ["c1dtlz1", "dc2dtlz3", "mw1"])
def test_problem_selection_constrained_many_defaults(name: str):
    selection = make_problem_selection(name)
    assert selection.n_var > 0
    assert selection.n_obj >= 2


@pytest.mark.parametrize("name", ["zcat1", "zcat14", "zcat20"])
def test_problem_selection_zcat_defaults(name: str):
    selection = make_problem_selection(name)
    assert selection.n_var == 30
    assert selection.n_obj == 2


def test_problem_selection_zcat_n_obj_override():
    selection = make_problem_selection("zcat6", n_var=40, n_obj=4)
    assert selection.n_var == 40
    assert selection.n_obj == 4


def test_tsplib_selection_encoding():
    selection = make_problem_selection("kroa100")
    assert selection.n_var == 100
    assert selection.n_obj == 2
    problem = selection.instantiate()
    assert getattr(problem, "encoding", "") == "permutation"


def test_invalid_problem_raises():
    with pytest.raises(KeyError):
        make_problem_selection("does_not_exist")
