import numpy as np

from vamos.algorithm.archive import CrowdingArchive, _single_front_crowding


def _dom(a, b):
    if np.all(a <= b) and np.any(a < b):
        return 1
    if np.all(b <= a) and np.any(b < a):
        return -1
    return 0


def test_crowding_archive_trims_by_crowding():
    archive = CrowdingArchive(capacity=2, dominance_fn=_dom, crowding_fn=_single_front_crowding)
    archive.add(np.array([0.0, 0.0]), np.array([0.0, 1.0]))
    archive.add(np.array([1.0, 1.0]), np.array([1.0, 0.0]))
    archive.add(np.array([0.5, 0.5]), np.array([0.5, 0.5]))
    X, F = archive.get_solutions()
    assert F.shape[0] == 2
    # Middle point should be trimmed (lowest crowding)
    assert not np.any(np.all(F == np.array([0.5, 0.5]), axis=1))
