from pathlib import Path

import pytest

import vamos
from vamos.experiment.runner import run_experiment
from vamos.foundation.core.experiment_config import ExperimentConfig


@pytest.mark.smoke
def test_run_experiment_creates_standard_layout(tmp_path):
    cfg = ExperimentConfig(output_root=str(tmp_path), population_size=6, max_evaluations=20, seed=1)
    metrics = run_experiment(
        algorithm="nsgaii",
        problem="zdt1",
        engine="numpy",
        config=cfg,
        selection_pressure=2,
    )

    run_dir = Path(metrics["output_dir"])
    assert run_dir.exists()
    assert {path.name for path in run_dir.iterdir()} == {"manifest.json", "result.npz", "environment.json"}
    manifest = vamos.load_run(run_dir, verify="all").manifest
    assert manifest.resolved_spec["problem"]["component_id"] == "vamos.problem:zdt1@1"
    assert manifest.resolved_spec["algorithm"]["component_id"] == "vamos.algorithm:nsgaii@1"
    assert manifest.resolved_spec["backend"]["kernel"]["component_id"] == "vamos.kernel:numpy@1"
