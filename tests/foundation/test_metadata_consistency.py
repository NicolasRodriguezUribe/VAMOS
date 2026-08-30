from pathlib import Path

import vamos
from vamos.experiment.runner import run_single
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.problem.registry import make_problem_selection


def test_manifest_and_result_bundle_are_consistent(tmp_path):
    selection = make_problem_selection("zdt1", n_var=6)
    cfg = ExperimentConfig(
        population_size=6, offspring_population_size=6, max_evaluations=20, seed=7, output_root=str(tmp_path / "results")
    )
    metrics = run_single("numpy", "nsgaii", selection, cfg)
    out_dir = Path(metrics["output_dir"])
    run = vamos.load_run(out_dir, verify="all")
    resolved = run.manifest.resolved_spec

    assert resolved["algorithm"]["component_id"] == "vamos.algorithm:nsgaii@1"
    assert resolved["backend"]["kernel"]["component_id"] == "vamos.kernel:numpy@1"
    assert resolved["problem"]["component_id"] == "vamos.problem:zdt1@1"
    assert resolved["seed"] == 7
    assert resolved["population"]["initial_size"] == cfg.population_size
    assert run.result.F is not None and run.result.F.shape[0] > 0
