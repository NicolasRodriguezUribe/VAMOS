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

    def crowding_distance(self, F: np.ndarray, fronts: list[list[int]] | None) -> np.ndarray:
        self.calls.append("crowding_distance")
        return NumbaKernel().nsga2_ranking(np.asarray(F, dtype=np.float64))[1]

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


class _FakeCppBackend:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate_offspring(
        self,
        X: np.ndarray,
        F: np.ndarray,
        params: dict[str, object],
        rng: np.random.Generator,
        xl: np.ndarray,
        xu: np.ndarray,
        n_offspring: int | None = None,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        self.calls.append("generate_offspring")
        count = int(n_offspring) if n_offspring is not None else X.shape[0]
        child = np.full((count, X.shape[1]), 0.25, dtype=np.float64)
        if out is not None:
            out[:] = child
            return out
        return child


class _FailingFakeCppBackend(_FakeCppBackend):
    def generate_offspring(
        self,
        X: np.ndarray,
        F: np.ndarray,
        params: dict[str, object],
        rng: np.random.Generator,
        xl: np.ndarray,
        xu: np.ndarray,
        n_offspring: int | None = None,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        self.calls.append("generate_offspring")
        raise RuntimeError("cpp fallback probe")


def test_numba_mixed_ranking_matches_numba_on_biobjective_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("vamos.foundation.kernel.mixed_backend.NativeNsga2Bridge", _FakeNativeBridge)
    monkeypatch.setattr("vamos.foundation.kernel.mixed_backend.CppBackend", _FakeCppBackend)
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
    assert mixed._native.calls == ["fast_non_dominated_sort", "crowding_distance"]


def test_numba_mixed_ranking_falls_back_for_three_objectives(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("vamos.foundation.kernel.mixed_backend.NativeNsga2Bridge", _FakeNativeBridge)
    monkeypatch.setattr("vamos.foundation.kernel.mixed_backend.CppBackend", _FakeCppBackend)
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


def test_numba_mixed_ranking_falls_back_to_numba_crowding_when_native_crowding_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeNativeBridgeNoCrowding(_FakeNativeBridge):
        def crowding_distance(self, F: np.ndarray, fronts: list[list[int]] | None) -> np.ndarray:
            self.calls.append("crowding_distance")
            raise AttributeError("missing crowding kernel")

    monkeypatch.setattr("vamos.foundation.kernel.mixed_backend.NativeNsga2Bridge", _FakeNativeBridgeNoCrowding)
    monkeypatch.setattr("vamos.foundation.kernel.mixed_backend.CppBackend", _FakeCppBackend)
    from vamos.foundation.kernel.mixed_backend import NumbaMixedKernel

    mixed = NumbaMixedKernel()
    numba = NumbaKernel()
    rng = np.random.default_rng(19)
    F = rng.integers(0, 7, size=(28, 2)).astype(np.float64)

    ranks_mixed, crowd_mixed = mixed.nsga2_ranking(F)
    ranks_numba, crowd_numba = numba.nsga2_ranking(F)

    np.testing.assert_array_equal(ranks_mixed, ranks_numba)
    np.testing.assert_allclose(crowd_mixed, crowd_numba, rtol=0.0, atol=1e-12)
    assert mixed.used_native_for == ["nsga2_ranking"]
    assert mixed._native.calls == ["fast_non_dominated_sort", "crowding_distance"]


@pytest.mark.parametrize("return_indices", [False, True])
def test_numba_mixed_survival_matches_numba_on_biobjective_cases(
    monkeypatch: pytest.MonkeyPatch,
    return_indices: bool,
) -> None:
    monkeypatch.setattr("vamos.foundation.kernel.mixed_backend.NativeNsga2Bridge", _FakeNativeBridge)
    monkeypatch.setattr("vamos.foundation.kernel.mixed_backend.CppBackend", _FakeCppBackend)
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
    assert mixed.used_native_for == ["nsga2_survival"]
    assert mixed._native.calls == ["fast_non_dominated_sort", "crowding_distance"]


@pytest.mark.parametrize("return_indices", [False, True])
def test_numba_mixed_survival_falls_back_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    return_indices: bool,
) -> None:
    monkeypatch.setattr("vamos.foundation.kernel.mixed_backend.NativeNsga2Bridge", _FakeNativeBridge)
    monkeypatch.setattr("vamos.foundation.kernel.mixed_backend.CppBackend", _FakeCppBackend)
    from vamos.foundation.kernel.mixed_backend import NumbaMixedKernel

    mixed = NumbaMixedKernel()
    mixed._native_survival_enabled = False
    numba = NumbaKernel()
    rng = np.random.default_rng(29)
    X = rng.random((12, 4), dtype=np.float32)
    F = rng.random((12, 2), dtype=np.float32)
    X_off = rng.random((12, 4), dtype=np.float32)
    F_off = rng.random((12, 2), dtype=np.float32)

    actual = mixed.nsga2_survival(X, F, X_off, F_off, 12, return_indices=return_indices)
    expected = numba.nsga2_survival(X, F, X_off, F_off, 12, return_indices=return_indices)

    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(np.asarray(a), np.asarray(e), rtol=0.0, atol=1e-12)
    assert mixed.used_native_for == []


def test_numba_mixed_survival_falls_back_when_enabled_but_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("vamos.foundation.kernel.mixed_backend.NativeNsga2Bridge", _FakeNativeBridge)
    monkeypatch.setattr("vamos.foundation.kernel.mixed_backend.CppBackend", _FakeCppBackend)
    from vamos.foundation.kernel.mixed_backend import NumbaMixedKernel

    mixed = NumbaMixedKernel()
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


def test_numba_mixed_real_native_survival_matches_numba() -> None:
    _require_native_available()
    from vamos.foundation.kernel.mixed_backend import NumbaMixedKernel

    mixed = NumbaMixedKernel()
    numba = NumbaKernel()
    rng = np.random.default_rng(131)
    X = rng.random((20, 4), dtype=np.float64)
    F = rng.random((20, 2), dtype=np.float64)
    X_off = rng.random((20, 4), dtype=np.float64)
    F_off = rng.random((20, 2), dtype=np.float64)

    actual = mixed.nsga2_survival(X, F, X_off, F_off, 20, return_indices=True)
    expected = numba.nsga2_survival(X, F, X_off, F_off, 20, return_indices=True)

    for a, e in zip(actual, expected):
        np.testing.assert_allclose(np.asarray(a), np.asarray(e), rtol=0.0, atol=1e-12)


def test_numba_mixed_generate_offspring_uses_cpp_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("vamos.foundation.kernel.mixed_backend.NativeNsga2Bridge", _FakeNativeBridge)
    monkeypatch.setattr("vamos.foundation.kernel.mixed_backend.CppBackend", _FakeCppBackend)
    from vamos.foundation.kernel.mixed_backend import NumbaMixedKernel

    mixed = NumbaMixedKernel()
    rng = np.random.default_rng(41)
    X = rng.random((6, 4), dtype=np.float64)
    F = rng.random((6, 2), dtype=np.float64)
    params = {
        "selection_pressure": 2,
        "crossover": {"prob": 0.9, "eta": 20.0},
        "mutation": {"prob": 0.25, "eta": 20.0},
    }

    offspring = mixed.generate_offspring(X, F, params, rng, np.zeros(4), np.ones(4), n_offspring=4)

    assert mixed.used_native_for == ["generate_offspring"]
    assert mixed._cpp.calls == ["generate_offspring"]
    np.testing.assert_allclose(offspring, np.full((4, 4), 0.25, dtype=np.float64), rtol=0.0, atol=1e-12)


def test_numba_mixed_generate_offspring_falls_back_from_cpp_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("vamos.foundation.kernel.mixed_backend.NativeNsga2Bridge", _FakeNativeBridge)
    monkeypatch.setattr("vamos.foundation.kernel.mixed_backend.CppBackend", _FailingFakeCppBackend)
    from vamos.foundation.kernel.mixed_backend import NumbaMixedKernel

    mixed = NumbaMixedKernel()
    rng = np.random.default_rng(43)
    X = rng.random((6, 4), dtype=np.float64)
    F = rng.random((6, 2), dtype=np.float64)
    params = {
        "selection_pressure": 2,
        "crossover": {"prob": 0.9, "eta": 20.0},
        "mutation": {"prob": 0.25, "eta": 20.0},
    }

    offspring = mixed.generate_offspring(X, F, params, rng, np.zeros(4), np.ones(4), n_offspring=4)

    assert offspring.shape == (4, 4)
    assert mixed._cpp.calls == ["generate_offspring"]
    assert "generate_offspring" not in mixed.used_native_for


def test_numba_mixed_capabilities_include_native_tags() -> None:
    _require_native_available()
    from vamos.foundation.kernel.mixed_backend import NumbaMixedKernel

    backend = NumbaMixedKernel()
    caps = set(backend.capabilities())
    assert {"numba", "native", "native:nsga2", "native:rank2d"}.issubset(caps)
