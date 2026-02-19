from vamos.foundation.problem.registry_info import get_problem_info, list_problems


def test_problem_zoo_lists_known_problem():
    infos = list_problems()
    names = {info.name for info in infos}
    assert "zdt1" in names
    info = get_problem_info("zdt1")
    assert info is not None
    assert info.name == "zdt1"
