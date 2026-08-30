from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import vamos
import vamos.ux.api as ux_api
from vamos.experiment.artifacts import (
    CompatibilityReport,
    IncompleteRunMetadataError,
    LoadLimits,
    ReplayReport,
    RunManifest,
    StoredRun,
    VerificationReport,
)
from vamos.experiment.optimization_result import OptimizationResult


def _explicit_context(seed: int = 4) -> tuple[dict[str, object], dict[str, object]]:
    source = json.loads(Path("docs/dev/run_artifact_examples/custom-manual/manifest.json").read_text(encoding="utf-8"))
    requested = source["requested_spec"]
    resolved = source["resolved_spec"]
    resolved["seed"] = seed
    return requested, resolved


def test_public_exports_and_signatures_are_canonical() -> None:
    assert not hasattr(ux_api, "save_result")
    assert vamos.LoadLimits is LoadLimits
    assert vamos.RunManifest is RunManifest
    assert vamos.StoredRun is StoredRun
    assert vamos.IncompleteRunMetadataError is IncompleteRunMetadataError
    assert vamos.CompatibilityReport is CompatibilityReport
    assert vamos.VerificationReport is VerificationReport
    assert vamos.ReplayReport is ReplayReport
    assert list(inspect.signature(vamos.save_result).parameters) == [
        "result",
        "path",
        "requested_spec",
        "resolved_spec",
        "labels",
        "limits",
    ]
    assert list(inspect.signature(vamos.load_run).parameters) == ["path", "verify", "limits"]
    assert list(inspect.signature(vamos.load_result).parameters) == ["path", "verify", "limits"]
    assert list(inspect.signature(vamos.verify_run).parameters) == ["path", "require_level", "limits"]
    assert list(inspect.signature(vamos.reproduce).parameters) == ["path", "output", "limits"]


def test_manual_result_requires_complete_execution_context(tmp_path: Path) -> None:
    result = OptimizationResult({"F": np.array([[1.0, 2.0]]), "X": np.array([[3, 4]])})

    with pytest.raises(IncompleteRunMetadataError, match="both requested_spec and resolved_spec"):
        vamos.save_result(result, tmp_path / "missing")
    requested, _ = _explicit_context()
    with pytest.raises(IncompleteRunMetadataError):
        vamos.save_result(result, tmp_path / "partial", requested_spec=requested)


def test_manual_result_with_explicit_complete_context_succeeds(tmp_path: Path) -> None:
    result = OptimizationResult(
        {
            "F": np.array([[1.0, 2.0]], dtype=np.float32),
            "X": np.array([[3, 4]], dtype=np.int16),
        }
    )
    requested, resolved = _explicit_context(seed=0)

    stored = vamos.save_result(
        result,
        tmp_path / "manual",
        requested_spec=requested,
        resolved_spec=resolved,
    )
    loaded = vamos.load_result(stored.root)

    assert stored.manifest["replayability"]["declared_level"] == "manual"
    assert stored.manifest.resolved_spec["seed"] == 0
    assert np.array_equal(loaded.F, result.F)
    assert np.array_equal(loaded.X, result.X)
    assert {path.name for path in stored.root.iterdir()} == {"manifest.json", "result.npz", "environment.json"}


def test_loading_is_data_only_and_does_not_call_optimize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = vamos.optimize("zdt1", pop_size=4, max_evaluations=4, seed=4)
    stored = vamos.save_result(result, tmp_path / "run")

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("optimization must not execute during load")

    monkeypatch.setattr(vamos, "optimize", forbidden)
    loaded_run = vamos.load_run(stored.root)
    loaded_result = vamos.load_result(stored.root)

    assert loaded_run.status == "succeeded"
    assert np.array_equal(loaded_result.F, result.F)


def test_custom_manual_contract_example_loads_without_component_resolution() -> None:
    run = vamos.load_run(Path("docs/dev/run_artifact_examples/custom-manual"))

    assert run.status == "succeeded"
    assert run.manifest["replayability"]["declared_level"] == "manual"
    assert run.result.F is not None
