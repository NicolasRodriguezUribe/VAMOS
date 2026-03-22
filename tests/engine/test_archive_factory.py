from __future__ import annotations

import numpy as np

from vamos.engine.algorithm.components.archive import HypervolumeArchive, UnboundedArchive
from vamos.engine.archive import ExternalArchiveConfig
from vamos.engine.archive.factory import setup_archive
from vamos.foundation.kernel.numpy_backend import NumPyKernel


def test_setup_archive_honors_extended_external_archive_config():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [2.0, 2.0]], dtype=float)
    F = np.array([[0.0, 2.0], [1.0, 1.0], [1.0, 1.0], [2.0, 0.0]], dtype=float)
    cfg = ExternalArchiveConfig(
        capacity=4,
        pruning="hv",
        hv_ref_point=[3.0, 3.0],
        truncate_size=3,
        deduplicate_in="both",
        objective_tolerance=1e-6,
        decision_tolerance=1e-6,
    )

    archive_X, archive_F, manager = setup_archive(NumPyKernel(), X, F, 2, 2, X.dtype, cfg)

    assert isinstance(manager, HypervolumeArchive)
    assert manager.truncate_size == 3
    assert np.array_equal(manager._fixed_ref, np.array([3.0, 3.0]))
    assert manager._deduplicate_in == "both"
    assert archive_X is not None and archive_F is not None
    assert archive_F.shape[0] == 3


def test_setup_archive_uses_shared_unbounded_archive_manager():
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    F = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    cfg = ExternalArchiveConfig(capacity=None, deduplicate_in="decision")

    archive_X, archive_F, manager = setup_archive(NumPyKernel(), X, F, 2, 2, X.dtype, cfg)

    assert isinstance(manager, UnboundedArchive)
    assert archive_X is not None and archive_F is not None
    assert archive_F.shape == (2, 2)
