from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vamos.foundation.kernel.native_bridge import NativeNsga2Bridge


def test_native_bridge_requires_native_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = SimpleNamespace(
        is_native_backend=lambda: False,
        backend_info=lambda: {"backend": "python-fallback", "native": False},
    )
    monkeypatch.setattr("vamos.foundation.kernel.native_bridge.import_module", lambda name: fake_module)
    bridge = NativeNsga2Bridge()
    with pytest.raises(ImportError, match="requires a native vamospp build"):
        bridge.require_native()


def test_native_bridge_fast_non_dominated_sort_normalizes_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    def fast_non_dominated_sort(F: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
        observed["dtype"] = F.dtype
        observed["c_contiguous"] = F.flags.c_contiguous
        observed["shape"] = F.shape
        return [[0, 1], [2]], np.array([0, 0, 1], dtype=np.int64)

    fake_module = SimpleNamespace(
        is_native_backend=lambda: True,
        backend_info=lambda: {"backend": "fake-native", "native": True},
        fast_non_dominated_sort=fast_non_dominated_sort,
    )
    monkeypatch.setattr("vamos.foundation.kernel.native_bridge.import_module", lambda name: fake_module)

    bridge = NativeNsga2Bridge()
    F = np.arange(12, dtype=np.float32).reshape(3, 4)[:, :2]
    fronts, ranks = bridge.fast_non_dominated_sort(F)

    assert fronts == [[0, 1], [2]]
    assert ranks.dtype == np.int64
    assert tuple(observed["shape"]) == (3, 2)
    assert observed["dtype"] == np.float64
    assert observed["c_contiguous"] is True


def test_native_bridge_nsga2_survival_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    def nsga2_survival(
        X: np.ndarray,
        F: np.ndarray,
        X_off: np.ndarray,
        F_off: np.ndarray,
        pop_size: int,
        return_indices: bool,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_comb = np.vstack((X, X_off))
        F_comb = np.vstack((F, F_off))
        idx = np.arange(pop_size, dtype=np.int64)
        if return_indices:
            return X_comb[idx], F_comb[idx], idx
        return X_comb[idx], F_comb[idx]

    fake_module = SimpleNamespace(
        is_native_backend=lambda: True,
        backend_info=lambda: {"backend": "fake-native", "native": True},
        nsga2_survival=nsga2_survival,
    )
    monkeypatch.setattr("vamos.foundation.kernel.native_bridge.import_module", lambda name: fake_module)

    bridge = NativeNsga2Bridge()
    X = np.arange(12, dtype=np.float32).reshape(3, 4)
    F = np.arange(6, dtype=np.float32).reshape(3, 2)
    X_off = X + 1
    F_off = F + 1

    X_new, F_new, idx = bridge.nsga2_survival(X, F, X_off, F_off, 3, return_indices=True)
    assert X_new.dtype == np.float64
    assert F_new.dtype == np.float64
    assert idx.dtype == np.int64
    assert X_new.flags.c_contiguous
    assert F_new.flags.c_contiguous


def test_native_bridge_crowding_distance_normalizes_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    def crowding_distance(F: np.ndarray, fronts: list[list[int]] | None) -> np.ndarray:
        observed["dtype"] = F.dtype
        observed["c_contiguous"] = F.flags.c_contiguous
        observed["shape"] = F.shape
        observed["fronts"] = fronts
        return np.array([np.inf, 1.5, np.inf], dtype=np.float64)

    fake_module = SimpleNamespace(
        is_native_backend=lambda: True,
        backend_info=lambda: {"backend": "fake-native", "native": True},
        crowding_distance=crowding_distance,
    )
    monkeypatch.setattr("vamos.foundation.kernel.native_bridge.import_module", lambda name: fake_module)

    bridge = NativeNsga2Bridge()
    F = np.arange(12, dtype=np.float32).reshape(3, 4)[:, :2]
    crowd = bridge.crowding_distance(F, [[0, 2], [1]])

    assert crowd.dtype == np.float64
    assert crowd.flags.c_contiguous
    assert tuple(observed["shape"]) == (3, 2)
    assert observed["dtype"] == np.float64
    assert observed["c_contiguous"] is True
    assert observed["fronts"] == [[0, 2], [1]]

