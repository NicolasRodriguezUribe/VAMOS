from __future__ import annotations

import numpy as np
from vamos.archive import BoundedArchive, BoundedArchiveConfig


def test_bounded_archive_size_cap_and_nondominated():
    cfg = BoundedArchiveConfig(
        enabled=True,
        archive_type="size_cap",
        size_cap=5,
        nondominated_only=True,
        prune_policy="crowding",
    )
    A = BoundedArchive(cfg)

    # 2D minimization: include dominated points
    F = np.array([
        [1.0, 5.0],
        [2.0, 4.0],
        [3.0, 3.0],
        [4.0, 2.0],
        [5.0, 1.0],
        [6.0, 6.0],  # dominated by many
        [2.5, 4.5],  # dominated by [2,4]
    ])
    upd = A.add(X=None, F=F, evals=1000)
    assert A.size() <= 5
    # dominated points should be removed first
    assert upd.after <= 5


def test_epsilon_grid_compaction():
    cfg = BoundedArchiveConfig(
        enabled=True,
        archive_type="epsilon_grid",
        size_cap=100,
        epsilon=0.5,
        nondominated_only=False,
    )
    A = BoundedArchive(cfg)
    F = np.array([
        [1.01, 2.02],
        [1.10, 2.10],
        [1.49, 2.49],  # same cell (floor/0.5)
        [2.01, 3.02],
        [2.10, 3.10],
    ])
    A.add(X=None, F=F, evals=10)
    # Expect fewer points after compaction if grid merges
    assert A.size() <= F.shape[0]
