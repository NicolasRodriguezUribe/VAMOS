import numpy as np

from vamos.engine.algorithm.components.archive import HypervolumeArchive, UnboundedArchive


def test_hypervolume_archive_trims_by_contribution():
    archive = HypervolumeArchive(capacity=2, n_var=2, n_obj=2, dtype=float)
    X = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    F = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    archive.update(X, F)
    result_X, result_F = archive.contents()
    assert result_F.shape[0] == 2
    # Middle point should be trimmed (lowest HV contribution)
    assert not np.any(np.all(result_F == np.array([0.5, 0.5]), axis=1))


def test_unbounded_archive_keeps_all_nondominated_points():
    archive = UnboundedArchive(n_var=1, n_obj=2, dtype=float)
    n = 12
    X = np.arange(n, dtype=float).reshape(-1, 1)
    F = np.column_stack([np.arange(n, dtype=float), np.arange(n - 1, -1, -1, dtype=float)])
    archive.update(X, F)
    _, result_F = archive.contents()
    assert result_F.shape == (n, 2)

    # Dominated point should not be added.
    archive.update(np.array([[999.0]]), np.array([[float(n + 1), float(n + 1)]]))
    _, result_F2 = archive.contents()
    assert result_F2.shape == (n, 2)
