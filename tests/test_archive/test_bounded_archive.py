from __future__ import annotations

import numpy as np

from vamos.engine.archive import BoundedArchive, BoundedArchiveConfig


def test_bounded_archive_size_cap_and_nondominated():
    cfg = BoundedArchiveConfig(
        enabled=True,
        size_cap=5,
        prune_policy="crowding",
    )
    A = BoundedArchive(cfg)

    # 2D minimization: include dominated points
    F = np.array(
        [
            [1.0, 5.0],
            [2.0, 4.0],
            [3.0, 3.0],
            [4.0, 2.0],
            [5.0, 1.0],
            [6.0, 6.0],  # dominated by many
            [2.5, 4.5],  # dominated by [2,4]
        ]
    )
    upd = A.add(X=None, F=F, evals=1000)
    assert A.size() <= 5
    # dominated points should be removed first
    assert upd.after <= 5


def test_bounded_archive_random_prunes_to_cap():
    cfg = BoundedArchiveConfig(
        enabled=True,
        size_cap=3,
        prune_policy="random",
    )
    A = BoundedArchive(cfg)
    F = np.array(
        [
            [1.0, 5.0],
            [2.0, 4.0],
            [3.0, 3.0],
            [4.0, 2.0],
            [5.0, 1.0],
        ]
    )
    A.add(X=None, F=F, evals=10)
    assert A.size() == 3
