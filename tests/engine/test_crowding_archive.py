import numpy as np

from vamos.engine.algorithm.components.archive import HypervolumeArchive, _single_front_crowding


def test_hypervolume_archive_trims_by_contribution():
    archive = HypervolumeArchive(capacity=2, n_var=2, n_obj=2, dtype=float)
    X = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    F = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    archive.update(X, F)
    result_X, result_F = archive.contents()
    assert result_F.shape[0] == 2
    # Middle point should be trimmed (lowest HV contribution)
    assert not np.any(np.all(result_F == np.array([0.5, 0.5]), axis=1))
