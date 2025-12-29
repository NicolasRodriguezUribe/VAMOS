import csv
import json
from pathlib import Path

from vamos.experiment.runner import run_experiment
from vamos.foundation.core.experiment_config import ExperimentConfig


TRACE_HEADER = [
    "step",
    "mating_id",
    "op_id",
    "op_name",
    "reward",
    "reward_survival",
    "reward_nd_insertions",
    "reward_hv_delta",
    "batch_size",
]

SUMMARY_HEADER = [
    "op_id",
    "op_name",
    "pulls",
    "mean_reward",
    "total_reward",
    "usage_fraction",
]


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = list(reader)
    return header, rows


def _base_config(output_root: str) -> ExperimentConfig:
    return ExperimentConfig(output_root=output_root, population_size=6, max_evaluations=30, seed=1)


def _aos_variation() -> dict:
    return {
        "adaptive_operator_selection": {
            "enabled": True,
            "policy": "eps_greedy",
            "reward_scope": "survival",
            "exploration": 0.2,
            "min_usage": 1,
            "rng_seed": 0,
            "operator_pool": [
                {
                    "crossover": ("sbx", {"prob": 0.9, "eta": 15.0}),
                    "mutation": ("pm", {"prob": "1/n", "eta": 20.0}),
                },
                {
                    "crossover": ("blx_alpha", {"prob": 0.9, "alpha": 0.5}),
                    "mutation": ("gaussian", {"prob": "1/n", "sigma": 0.1}),
                },
            ],
        }
    }


def test_aos_artifacts_written_and_registered(tmp_path):
    cfg = _base_config(str(tmp_path))
    metrics = run_experiment(
        algorithm="nsgaii",
        problem="zdt1",
        engine="numpy",
        config=cfg,
        selection_pressure=2,
        nsgaii_variation=_aos_variation(),
    )
    run_dir = Path(metrics["output_dir"])
    trace_path = run_dir / "aos_trace.csv"
    summary_path = run_dir / "aos_summary.csv"
    assert trace_path.is_file()
    assert summary_path.is_file()

    trace_header, trace_rows = _read_csv(trace_path)
    summary_header, summary_rows = _read_csv(summary_path)
    assert trace_header == TRACE_HEADER
    assert summary_header == SUMMARY_HEADER
    assert len(trace_rows) >= 1
    assert len(summary_rows) >= 1

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    artifacts = metadata.get("artifacts", {})
    assert artifacts.get("aos_trace") == "aos_trace.csv"
    assert artifacts.get("aos_summary") == "aos_summary.csv"

    resolved = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    assert "adaptive_operator_selection" in resolved


def test_aos_disabled_does_not_write_artifacts(tmp_path):
    cfg = _base_config(str(tmp_path))
    metrics = run_experiment(
        algorithm="nsgaii",
        problem="zdt1",
        engine="numpy",
        config=cfg,
        selection_pressure=2,
    )
    run_dir = Path(metrics["output_dir"])
    assert not (run_dir / "aos_trace.csv").exists()
    assert not (run_dir / "aos_summary.csv").exists()

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    artifacts = metadata.get("artifacts", {})
    assert "aos_trace" not in artifacts
    assert "aos_summary" not in artifacts
