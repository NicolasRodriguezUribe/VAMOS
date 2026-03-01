from __future__ import annotations

import numpy as np
import pytest

from vamos.foundation.kernel.native_bridge import NativeNsga2Bridge
from vamos.foundation.kernel.numba_backend import NumbaKernel
from vamos.foundation.kernel.numpy_backend import _fast_non_dominated_sort


class _FakeNativeBridge:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def require_native(self) -> None:
        return None

    def available(self) -> bool:
        return True

    @staticmethod
    def _as_float64_c(value: np.ndarray, *, name: str, ndim: int) -> np.ndarray:
        arr = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
        if arr.ndim != ndim:
            raise ValueError(f"{name} must be {ndim}D.")
        return arr

    def fast_non_dominated_sort(self, F: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
        self.calls.append("fast_non_dominated_sort")
        fronts, ranks = _fast_non_dominated_sort(np.asarray(F, dtype=np.float64))
        return fronts, np.ascontiguousarray(ranks.astype(np.int64))

    def nsga2_survival(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("native survival should not be used in phase 1")


class _FakeNativeBridgeWithSurvival(_FakeNativeBridge):
    def nsga2_survival(
        self,
        X: np.ndarray,
        F: np.ndarray,
        X_off: np.ndarray,
        F_off: np.ndarray,
        pop_size: int,
        *,
        return_indices: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.calls.append("nsga2_survival")
        X_new = np.ascontiguousarray(X_off[:pop_size], dtype=np.float64)
        F_new = np.ascontiguousarray(F_off[:pop_size], dtype=np.float64)
        if return_indices:
            idx = np.arange(pop_size, dtype=np.int64)
            return X_new, F_new, idx
        return X_new, F_new


def test_numba_mixed_ranking_matches_numba_on_biobjective_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("vamos.foundation.kernel.mixed_backend.NativeNsga2Bridge", _FakeNativeBridge)
    from vamos.foundation.kernel.mixed_backend import NumbaMixedKernel

    mixed = NumbaMixedKernel()
    numba = NumbaKernel()
    rng = np.random.default_rng(7)
    F = rng.integers(0, 9, size=(32, 2)).astype(np.float32)

    ranks_mixed, crowd_mixed = mixed.nsga2_ranking(F)
    ranks_numba, crowd_numba = numba.nsga2_ranking(F.astype(np.float64))

    np.testing.assert_array_equal(ranks_mixed, ranks_numba)
    np.testing.assert_allclose(crowd_mixed, crowd_numba, rtol=0.0, atol=1e-12)
    assert mixed.used_native_for == ["nsga2_ranking"]


def test_numba_mixed_ranking_falls_back_for_three_objectives(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("vamos.foundation.kernel.mixed_backend.NativeNsga2Bridge", _FakeNativeBridge)
    from vamos.foundation.kernel.mixed_backend import NumbaMixedKernel

    mixed = NumbaMixedKernel()
    numba = NumbaKernel()
    rng = np.random.default_rng(11)
    F = rng.random((24, 3), dtype=np.float64)

    ranks_mixed, crowd_mixed = mixed.nsga2_ranking(F)
    ranks_numba, crowd_numba = numba.nsga2_ranking(F)

    np.testing.assert_array_equal(ranks_mixed, ranks_numba)
    np.testing.assert_allclose(crowd_mixed, crowd_numba, rtol=0.0, atol=1e-12)
    assert mixed.used_native_for == []


@pytest.mark.parametrize("return_indices", [False, True])
def test_numba_mixed_survival_delegates_to_numba(monkeypatch: pytest.MonkeyPatch, return_indices: bool) -> None:
    monkeypatch.setattr("vamos.foundation.kernel.mixed_backend.NativeNsga2Bridge", _FakeNativeBridge)
    from vamos.foundation.kernel.mixed_backend import NumbaMixedKernel

    mixed = NumbaMixedKernel()
    numba = NumbaKernel()
    rng = np.random.default_rng(23)
    X = rng.random((12, 4), dtype=np.float64)
    F = rng.random((12, 2), dtype=np.float64)
    X_off = rng.random((12, 4), dtype=np.float64)
    F_off = rng.random((12, 2), dtype=np.float64)

    actual = mixed.nsga2_survival(X, F, X_off, F_off, 12, return_indices=return_indices)
    expected = numba.nsga2_survival(X, F, X_off, F_off, 12, return_indices=return_indices)

    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(np.asarray(a), np.asarray(e), rtol=0.0, atol=1e-12)
    assert mixed.used_native_for == []


@pytest.mark.parametrize("return_indices", [False, True])
def test_numba_mixed_survival_can_use_native_when_explicitly_enabled(
    monkeypatch: pytest.MonkeyPatch,
    return_indices: bool,
) -> None:
    monkeypatch.setattr("vamos.foundation.kernel.mixed_backend.NativeNsga2Bridge", _FakeNativeBridgeWithSurvival)
    from vamos.foundation.kernel.mixed_backend import NumbaMixedKernel

    mixed = NumbaMixedKernel()
    mixed._native_survival_enabled = True
    rng = np.random.default_rng(29)
    X = rng.random((12, 4), dtype=np.float32)
    F = rng.random((12, 2), dtype=np.float32)
    X_off = rng.random((12, 4), dtype=np.float32)
    F_off = rng.random((12, 2), dtype=np.float32)

    actual = mixed.nsga2_survival(X, F, X_off, F_off, 12, return_indices=return_indices)

    assert mixed.used_native_for == ["nsga2_survival"]
    np.testing.assert_allclose(np.asarray(actual[0]), np.asarray(X_off, dtype=np.float64), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(actual[1]), np.asarray(F_off, dtype=np.float64), rtol=0.0, atol=1e-12)
    if return_indices:
        np.testing.assert_array_equal(np.asarray(actual[2]), np.arange(12, dtype=np.int64))


def test_numba_mixed_survival_falls_back_when_enabled_but_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("vamos.foundation.kernel.mixed_backend.NativeNsga2Bridge", _FakeNativeBridgeWithSurvival)
    from vamos.foundation.kernel.mixed_backend import NumbaMixedKernel

    mixed = NumbaMixedKernel()
    mixed._native_survival_enabled = True
    numba = NumbaKernel()
    rng = np.random.default_rng(31)
    X = rng.random((10, 4), dtype=np.float64)
    F = rng.random((10, 3), dtype=np.float64)
    X_off = rng.random((10, 4), dtype=np.float64)
    F_off = rng.random((10, 3), dtype=np.float64)

    actual = mixed.nsga2_survival(X, F, X_off, F_off, 10, return_indices=True)
    expected = numba.nsga2_survival(X, F, X_off, F_off, 10, return_indices=True)

    for a, e in zip(actual, expected):
        np.testing.assert_allclose(np.asarray(a), np.asarray(e), rtol=0.0, atol=1e-12)
    assert mixed.used_native_for == []


def _native_available() -> bool:
    return NativeNsga2Bridge().available()


def _require_native_available() -> None:
    if not _native_available():
        pytest.skip("native vamospp is not available")


def test_numba_mixed_real_native_ranking_matches_numba() -> None:
    _require_native_available()
    from vamos.foundation.kernel.mixed_backend import NumbaMixedKernel

    mixed = NumbaMixedKernel()
    numba = NumbaKernel()
    rng = np.random.default_rng(101)
    F = rng.integers(0, 8, size=(40, 2)).astype(np.float64)

    ranks_mixed, crowd_mixed = mixed.nsga2_ranking(F)
    ranks_numba, crowd_numba = numba.nsga2_ranking(F)

    np.testing.assert_array_equal(ranks_mixed, ranks_numba)
    np.testing.assert_allclose(crowd_mixed, crowd_numba, rtol=0.0, atol=1e-12)


def test_numba_mixed_capabilities_include_native_tags() -> None:
    _require_native_available()
    from vamos.foundation.kernel.mixed_backend import NumbaMixedKernel

    backend = NumbaMixedKernel()
    caps = set(backend.capabilities())
    assert {"numba", "native", "native:nsga2", "native:rank2d"}.issubset(caps)
