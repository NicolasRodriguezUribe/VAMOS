import json
from pathlib import Path

import numpy as np

from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.problem.registry import make_problem_selection
from vamos.experiment.runner import run_single


def test_metadata_and_resolved_config_are_consistent(tmp_path):
    selection = make_problem_selection("zdt1", n_var=6)
    cfg = ExperimentConfig(
        population_size=6, offspring_population_size=6, max_evaluations=20, seed=7, output_root=str(tmp_path / "results")
    )
    metrics = run_single("numpy", "nsgaii", selection, cfg)
    out_dir = Path(metrics["output_dir"])
    meta = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    resolved = json.loads((out_dir / "resolved_config.json").read_text(encoding="utf-8"))

    assert meta["algorithm"] == "nsgaii"
    assert meta["backend"] == "numpy"
    assert meta["seed"] == 7
    assert meta["problem"]["key"] == "zdt1"
    assert resolved["algorithm"] == "nsgaii"
    assert resolved["engine"] == "numpy"
    assert resolved["problem"] == "zdt1"
    assert resolved["seed"] == 7
    assert resolved["population_size"] == cfg.population_size
    fun = np.loadtxt(out_dir / "FUN.csv", delimiter=",")
    assert fun.shape[0] > 0
