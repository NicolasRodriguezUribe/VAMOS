import numpy as np

from vamos.engine.algorithm.ibea.helpers import epsilon_indicator


def test_epsilon_indicator_matches_jmetal_definition() -> None:
    F = np.array([[0.0, 0.0], [1.0, 2.0]])
    ind = epsilon_indicator(F)

    assert ind.shape == (2, 2)
    assert np.isclose(ind[0, 1], 2.0)
    assert np.isclose(ind[1, 0], -1.0)
