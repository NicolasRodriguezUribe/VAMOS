import numpy as np

from vamos.engine.algorithm.ibea.helpers import epsilon_indicator, hypervolume_indicator
from vamos.foundation.quality_indicators.hypervolume import hypervolume


def test_epsilon_indicator_matches_jmetal_definition() -> None:
    F = np.array([[0.0, 0.0], [1.0, 2.0]])
    ind = epsilon_indicator(F)

    assert ind.shape == (2, 2)
    assert np.isclose(ind[0, 1], 2.0)
    assert np.isclose(ind[1, 0], -1.0)


def _hypervolume_indicator_reference(F: np.ndarray) -> np.ndarray:
    ref = np.max(F, axis=0) + 1.0
    indicator = np.zeros((F.shape[0], F.shape[0]), dtype=float)
    for i in range(F.shape[0]):
        for j in range(F.shape[0]):
            if i == j:
                continue
            indicator[i, j] = hypervolume(F[j : j + 1], ref) - hypervolume(np.vstack([F[i], F[j]]), ref)
    return indicator


def test_hypervolume_indicator_matches_reference_pairwise_definition() -> None:
    F = np.array(
        [
            [0.3, 0.9],
            [0.6, 0.5],
            [0.8, 0.2],
            [0.4, 0.7],
        ],
        dtype=float,
    )

    expected = _hypervolume_indicator_reference(F)
    actual = hypervolume_indicator(F)

    np.testing.assert_allclose(actual, expected)
