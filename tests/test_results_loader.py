from pathlib import Path

from vamos.experiment.runner import run_experiment
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.ux.analysis.results import aggregate_results, discover_runs, load_run_data


def test_discover_and_load_results(tmp_path):
    cfg = ExperimentConfig(output_root=str(tmp_path), population_size=6, max_evaluations=20, seed=2)
    run_experiment(
        algorithm="nsgaii",
        problem="zdt1",
        engine="numpy",
        config=cfg,
        selection_pressure=2,
    )

    runs = discover_runs(tmp_path)
    assert runs, "Expected to find at least one run"

    run = runs[0]
    data = load_run_data(run)
    assert data.metadata["problem"]["key"] == "zdt1"
    assert data.F is not None and data.F.shape[1] == 2

    agg = aggregate_results(runs)
    if isinstance(agg, list):
        assert any(rec.get("algorithm") == "nsgaii" for rec in agg)
    else:
        assert "algorithm" in agg.columns

