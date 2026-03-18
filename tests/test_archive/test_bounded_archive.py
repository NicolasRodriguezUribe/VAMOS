from __future__ import annotations

import numpy as np
import pytest

from vamos.engine.archive import BoundedArchive, BoundedArchiveConfig, ExternalArchiveConfig
from vamos.engine.archive import bounded_archive as ba


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


def test_epsilon_grid_compaction():
    cfg = BoundedArchiveConfig(
        enabled=True,
        archive_type="epsilon_grid",
        size_cap=100,
        epsilon=0.5,
        nondominated_only=False,
    )
    A = BoundedArchive(cfg)
    F = np.array(
        [
            [1.01, 2.02],
            [1.10, 2.10],
            [1.49, 2.49],  # same cell (floor/0.5)
            [2.01, 3.02],
            [2.10, 3.10],
        ]
    )
    A.add(X=None, F=F, evals=10)
    # Expect fewer points after compaction if grid merges
    assert A.size() <= F.shape[0]


def test_legacy_prune_policy_aliases_are_rejected():
    with pytest.raises(ValueError, match="Unsupported prune_policy 'hv_contrib'"):
        ExternalArchiveConfig(capacity=10, pruning="hv_contrib")
    with pytest.raises(ValueError, match="Unsupported prune_policy 'mc_hv_contrib'"):
        ExternalArchiveConfig(capacity=10, pruning="mc_hv_contrib")
    with pytest.raises(ValueError, match="Unsupported prune_policy 'hv_contrib'"):
        BoundedArchiveConfig(prune_policy="hv_contrib")
    with pytest.raises(ValueError, match="Unsupported prune_policy 'mc_hv_contrib'"):
        BoundedArchiveConfig(prune_policy="mc_hv_contrib")


def test_spea2_prune_policy_name_is_rejected_in_favor_of_knn():
    with pytest.raises(ValueError, match="Unsupported prune_policy 'spea2'"):
        ExternalArchiveConfig(capacity=10, pruning="spea2")
    with pytest.raises(ValueError, match="Unsupported prune_policy 'spea2'"):
        BoundedArchiveConfig(prune_policy="spea2")


def test_random_prune_policy_name_is_rejected():
    with pytest.raises(ValueError, match="Unsupported prune_policy 'random'"):
        ExternalArchiveConfig(capacity=10, pruning="random")
    with pytest.raises(ValueError, match="Unsupported prune_policy 'random'"):
        BoundedArchiveConfig(prune_policy="random")


def test_knn_prune_policy_name_is_accepted():
    assert ExternalArchiveConfig(capacity=10, pruning="knn").pruning == "knn"
    assert BoundedArchiveConfig(prune_policy="knn").prune_policy == "knn"


def test_maxmin_and_ref_dirs_prune_policy_names_are_accepted():
    assert ExternalArchiveConfig(capacity=10, pruning="maxmin").pruning == "maxmin"
    assert ExternalArchiveConfig(capacity=10, pruning="ref_dirs").pruning == "ref_dirs"
    assert BoundedArchiveConfig(prune_policy="maxmin").prune_policy == "maxmin"
    assert BoundedArchiveConfig(prune_policy="ref_dirs").prune_policy == "ref_dirs"


def test_bounded_archive_maxmin_prunes_to_target_size():
    cfg = BoundedArchiveConfig(size_cap=3, prune_policy="maxmin", nondominated_only=True)
    archive = BoundedArchive(cfg)
    F = np.array(
        [
            [0.0, 10.0],
            [1.0, 9.0],
            [5.0, 5.0],
            [9.0, 1.0],
            [10.0, 0.0],
        ]
    )
    archive.add(X=None, F=F, evals=10)
    kept = archive.F
    assert kept.shape[0] == 3
    assert np.any(np.all(kept == np.array([0.0, 10.0]), axis=1))
    assert np.any(np.all(kept == np.array([10.0, 0.0]), axis=1))


def test_bounded_archive_ref_dirs_prunes_to_target_size():
    cfg = BoundedArchiveConfig(size_cap=3, prune_policy="ref_dirs", nondominated_only=True)
    archive = BoundedArchive(cfg)
    F = np.array(
        [
            [0.90, 0.05, 0.05],
            [0.05, 0.90, 0.05],
            [0.05, 0.05, 0.90],
            [0.34, 0.34, 0.34],
            [0.60, 0.20, 0.20],
        ]
    )
    archive.add(X=None, F=F, evals=10)
    kept = archive.F
    assert kept.shape[0] == 3
    assert np.any(np.all(kept == np.array([0.90, 0.05, 0.05]), axis=1))
    assert np.any(np.all(kept == np.array([0.05, 0.90, 0.05]), axis=1))
    assert np.any(np.all(kept == np.array([0.05, 0.05, 0.90]), axis=1))


def test_bounded_archive_hv_uses_moocore_for_many_objective_pruning(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    class _FakeMooCore:
        @staticmethod
        def hv_contributions(F: np.ndarray, *, ref: np.ndarray) -> np.ndarray:
            calls.append((F.copy(), ref.copy()))
            return np.array([0.4, 0.1, 0.3, 0.2], dtype=float)

    monkeypatch.setattr(ba, "_moocore", _FakeMooCore)

    cfg = BoundedArchiveConfig(size_cap=3, prune_policy="hv", nondominated_only=True, hv_ref_point=[2.0, 2.0, 2.0])
    archive = BoundedArchive(cfg)
    F = np.array(
        [
            [0.9, 0.2, 0.2],
            [0.2, 0.9, 0.2],
            [0.2, 0.2, 0.9],
            [0.45, 0.45, 0.45],
        ]
    )

    archive.add(X=None, F=F, evals=10)

    assert len(calls) == 1
    assert archive.F.shape[0] == 3
    assert not np.any(np.all(archive.F == F[1], axis=1))


def test_bounded_archive_mc_hv_keeps_monte_carlo_path_even_with_moocore(monkeypatch: pytest.MonkeyPatch):
    class _FakeMooCore:
        @staticmethod
        def hv_contributions(F: np.ndarray, *, ref: np.ndarray) -> np.ndarray:
            raise AssertionError("mc_hv should not call moocore.hv_contributions")

    monkeypatch.setattr(ba, "_moocore", _FakeMooCore)

    cfg = BoundedArchiveConfig(size_cap=3, prune_policy="mc_hv", nondominated_only=True, hv_ref_point=[2.0, 2.0, 2.0])
    archive = BoundedArchive(cfg)
    monkeypatch.setattr(
        archive,
        "_mc_hv_contrib_prune",
        lambda ref, k: np.array([2], dtype=int),
    )

    F = np.array(
        [
            [0.9, 0.2, 0.2],
            [0.2, 0.9, 0.2],
            [0.2, 0.2, 0.9],
            [0.45, 0.45, 0.45],
        ]
    )

    archive.add(X=None, F=F, evals=10)

    assert archive.F.shape[0] == 3
    assert not np.any(np.all(archive.F == F[2], axis=1))
