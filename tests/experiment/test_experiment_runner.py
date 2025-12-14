import json
from pathlib import Path

import pytest

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
    assert (run_dir / "FUN.csv").is_file()
    assert (run_dir / "metadata.json").is_file()

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["problem"]["key"] == "zdt1"
    assert metadata["algorithm"] == "nsgaii"
    assert metadata["backend"] == "numpy"
