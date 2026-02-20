import warnings

import numpy as np
import pytest

import vamos.engine.algorithm.components.archive as archive_mod
from vamos.engine.algorithm.components.archive import (
    CrowdingDistanceArchive,
    HypervolumeArchive,
    SPEA2Archive,
    UnboundedArchive,
)


def _tradeoff_front(n: int) -> np.ndarray:
    x = np.arange(n, dtype=float)
    y = (n - x).astype(float)
    return np.column_stack([x, y])


def test_hypervolume_archive_trims_by_contribution():
    archive = HypervolumeArchive(capacity=2, n_var=2, n_obj=2, dtype=float)
    X = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    F = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    archive.update(X, F)
    _, result_F = archive.contents()
    assert result_F.shape[0] == 2
    # Middle point should be trimmed (lowest HV contribution).
    assert not np.any(np.all(result_F == np.array([0.5, 0.5]), axis=1))


def test_hysteresis_truncate_size_behavior():
    archive = CrowdingDistanceArchive(capacity=20, truncate_size=10, n_var=1, n_obj=2, dtype=float)

    X0 = np.arange(25, dtype=float).reshape(-1, 1)
    F0 = _tradeoff_front(25)
    archive.update(X0, F0)
    assert archive.contents()[1].shape[0] == 10

    X1 = np.arange(25, 30, dtype=float).reshape(-1, 1)
    F1_x = np.arange(25, 30, dtype=float)
    F1 = np.column_stack([F1_x, 25.0 - F1_x])
    archive.update(X1, F1)
    assert archive.contents()[1].shape[0] == 15

    X2 = np.arange(30, 40, dtype=float).reshape(-1, 1)
    F2_x = np.arange(30, 40, dtype=float)
    F2 = np.column_stack([F2_x, 25.0 - F2_x])
    archive.update(X2, F2)
    assert archive.contents()[1].shape[0] == 10


def test_spea2_archive_truncates_to_target_size():
    archive = SPEA2Archive(capacity=5, n_var=1, n_obj=3, dtype=float)
    X = np.arange(12, dtype=float).reshape(-1, 1)
    F = np.column_stack(
        [
            np.linspace(0.0, 1.0, 12),
            np.linspace(1.0, 0.0, 12),
            np.abs(np.linspace(-0.5, 0.5, 12)),
        ]
    )
    archive.update(X, F)
    _, kept_F = archive.contents()
    assert kept_F.shape[0] == 5


def test_archive_feasibility_prefers_feasible_points():
    archive = CrowdingDistanceArchive(capacity=10, n_var=1, n_obj=2, dtype=float, n_con=1)
    X = np.array([[0.0], [1.0], [2.0]])
    F = np.array([[0.1, 0.1], [1.0, 1.0], [2.0, 2.0]])
    G = np.array([[1.0], [0.0], [0.0]])

    archive.update(X, F, G)
    _, kept_F = archive.contents()
    assert kept_F.shape[0] == 1
    np.testing.assert_allclose(kept_F[0], np.array([1.0, 1.0]))


def test_archive_feasibility_keeps_least_violating_when_no_feasible():
    archive = CrowdingDistanceArchive(capacity=10, n_var=1, n_obj=2, dtype=float, n_con=1)
    X = np.array([[0.0], [1.0], [2.0]])
    F = np.array([[0.0, 0.0], [1.0, 0.5], [0.5, 1.0]])
    G = np.array([[2.0], [1.0], [1.0]])

    archive.update(X, F, G)
    _, kept_F = archive.contents()
    assert kept_F.shape[0] == 2
    assert not np.any(np.all(kept_F == np.array([0.0, 0.0]), axis=1))


def test_hypervolume_reference_is_stable():
    archive = HypervolumeArchive(capacity=2, n_var=1, n_obj=2, dtype=float, ref_offset=1.0)
    X0 = np.array([[0.0], [1.0], [2.0]])
    F0 = np.array([[10.0, 1.0], [1.0, 10.0], [6.0, 6.0]])
    archive.update(X0, F0)
    assert archive._global_worst is not None  # internal monotone reference tracker
    first_ref = archive._global_worst.copy()

    X1 = np.array([[3.0]])
    F1 = np.array([[0.1, 0.1]])
    archive.update(X1, F1)
    assert archive._global_worst is not None
    assert np.all(archive._global_worst >= first_ref)


def test_decision_space_dedup_keeps_equal_objectives_with_distinct_decisions():
    archive = UnboundedArchive(
        n_var=2,
        n_obj=2,
        dtype=float,
        deduplicate_in="decision",
        decision_tolerance=1e-32,
    )
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    F = np.array([[1.0, 1.0], [1.0 + 1e-12, 1.0 - 1e-12]])
    archive.update(X, F)
    _, kept_F = archive.contents()
    assert kept_F.shape[0] == 2


def test_objective_space_dedup_collapses_equal_objectives():
    archive = UnboundedArchive(
        n_var=2,
        n_obj=2,
        dtype=float,
        deduplicate_in="objective",
        objective_tolerance=1e-10,
    )
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    F = np.array([[1.0, 1.0], [1.0 + 1e-12, 1.0 - 1e-12]])
    archive.update(X, F)
    _, kept_F = archive.contents()
    assert kept_F.shape[0] == 1


def test_hv_fallback_warns_once(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(archive_mod, "_moocore", None)
    monkeypatch.setattr(archive_mod, "_HV_FALLBACK_WARNED", False)
    F = np.array([[0.0, 1.0], [1.0, 0.0]])
    ref = np.array([2.0, 2.0])

    with pytest.warns(UserWarning, match="moocore"):
        archive_mod._hv_contributions(F, ref)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        archive_mod._hv_contributions(F, ref)
    assert len(rec) == 0


def test_nsgaii_constrained_with_external_archive():
    """Constrained NSGA-II with external archive must not crash on G shape mismatch.

    Regression test: ``tell_nsgaii`` used to pass ``combined_X/combined_F``
    (pop_size + offspring_size rows) to ``update_archives`` while ``G`` defaulted
    to ``state.G`` (pop_size rows after survival), causing a shape mismatch in the
    archive's feasibility filter.
    """
    from vamos.engine.algorithm.config import NSGAIIConfig
    from vamos.engine.algorithm.nsgaii import NSGAII
    from vamos.foundation.kernel.numpy_backend import NumPyKernel

    class _ConstrainedBiobj:
        n_var = 2
        n_obj = 2
        n_constr = 1
        xl = np.array([0.0, 0.0])
        xu = np.array([1.0, 1.0])
        encoding = "real"

        def evaluate(self, X, out):
            out["F"] = np.column_stack([X[:, 0], 1.0 - X[:, 0] + X[:, 1]])
            out["G"] = (X[:, 0] + X[:, 1] - 1.5)[:, None]

    pop = 12
    cfg = (
        NSGAIIConfig.builder()
        .pop_size(pop)
        .offspring_size(pop)
        .crossover("sbx", prob=0.9, eta=15.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", pressure=2)
        .constraint_mode("feasibility")
        .external_archive(capacity=pop * 2, pruning="crowding")
        .build()
    )

    algo = NSGAII(cfg.to_dict(), kernel=NumPyKernel())
    problem = _ConstrainedBiobj()
    # Must complete without ValueError from shape mismatch
    result = algo.run(problem, termination=("max_evaluations", pop * 4), seed=42)

    assert result["F"].shape[0] > 0
    assert "archive" in result
    assert result["archive"]["F"].shape[0] > 0
