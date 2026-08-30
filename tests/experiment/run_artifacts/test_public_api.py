from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

import vamos
from vamos.experiment.artifacts import LoadLimits, RunManifest, StoredRun
from vamos.experiment.optimization_result import OptimizationResult
from vamos.ux.api import save_result as ux_save_result


def test_public_exports_and_signatures_are_canonical() -> None:
    assert vamos.save_result is ux_save_result
    assert vamos.LoadLimits is LoadLimits
    assert vamos.RunManifest is RunManifest
    assert vamos.StoredRun is StoredRun
    assert list(inspect.signature(vamos.save_result).parameters) == [
        "result",
        "path",
        "requested_spec",
        "labels",
        "limits",
    ]
    assert list(inspect.signature(vamos.load_run).parameters) == ["path", "verify", "limits"]
    assert list(inspect.signature(vamos.load_result).parameters) == ["path", "verify", "limits"]


def test_existing_ux_save_result_call_succeeds_ra038(tmp_path: Path) -> None:
    result = OptimizationResult(
        {
            "F": np.array([[1.0, 2.0]], dtype=np.float32),
            "X": np.array([[3, 4]], dtype=np.int16),
        }
    )

    stored = ux_save_result(result, str(tmp_path / "historical-call"))
    loaded = vamos.load_result(tmp_path / "historical-call")

    assert isinstance(stored, StoredRun)
    assert stored.manifest["replayability"]["declared_level"] == "unavailable"
    assert np.array_equal(loaded.F, result.F)
    assert np.array_equal(loaded.X, result.X)
    assert (stored.root / "FUN.csv").is_file()
    assert (stored.root / "X.csv").is_file()


def test_historical_resultlike_with_only_f_and_x_remains_valid(tmp_path: Path) -> None:
    class HistoricalResultLike:
        F = np.array([[0.25, 0.75]], dtype=np.float64)
        X = np.array([[1, 0]], dtype=np.int8)

    result = HistoricalResultLike()
    ux_save_result(result, str(tmp_path / "minimal-resultlike"))
    loaded = vamos.load_result(tmp_path / "minimal-resultlike")

    assert np.array_equal(loaded.F, result.F)
    assert np.array_equal(loaded.X, result.X)


def test_loading_is_data_only_and_does_not_call_optimize_ra033(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = OptimizationResult({"F": np.array([[1.0, 2.0]]), "X": np.array([[0.5]])}, meta={"seed": 4})
    stored = vamos.save_result(result, tmp_path / "run")

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("optimization must not execute during load")

    monkeypatch.setattr(vamos, "optimize", forbidden)
    loaded_run = vamos.load_run(stored.root)
    loaded_result = vamos.load_result(stored.root)

    assert loaded_run.status == "succeeded"
    assert np.array_equal(loaded_result.F, result.F)


def test_custom_manual_contract_example_loads_without_component_resolution_ra033() -> None:
    run = vamos.load_run(Path("docs/dev/run_artifact_examples/custom-manual"))

    assert run.status == "succeeded"
    assert run.manifest["replayability"]["declared_level"] == "manual"
    assert run.result.F is not None
