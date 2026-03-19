import csv
import json
from pathlib import Path

from vamos.experiment.benchmark.archive_family import ARCHIVE_FAMILY_DEFAULT_ALGORITHMS
from vamos.experiment.benchmark.runner import run_benchmark_suite
from vamos.experiment.benchmark.suites import get_benchmark_suite, list_benchmark_suites


def test_archive_family_suites_are_registered():
    names = list_benchmark_suites()
    assert "NSGAII_archive_family_smoke" in names
    assert "NSGAII_archive_family_core" in names


def test_archive_family_smoke_benchmark_writes_summary_artifacts(tmp_path: Path):
    suite = get_benchmark_suite("NSGAII_archive_family_smoke")
    result = run_benchmark_suite(
        suite=suite,
        algorithms=None,
        metrics=None,
        base_output_dir=tmp_path,
        global_config_overrides={
            "engine": "numpy",
            "population_size": 8,
            "offspring_population_size": 8,
        },
    )

    summary_dir = tmp_path / "summary"
    metrics_path = summary_dir / "metrics.csv"
    runs_path = summary_dir / "archive_family_runs.csv"
    means_path = summary_dir / "archive_family_means.csv"
    summary_json = summary_dir / "archive_family_summary.json"

    assert result.summary_path == metrics_path
    assert metrics_path.exists()
    assert runs_path.exists()
    assert means_path.exists()
    assert summary_json.exists()

    with metrics_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows
    assert {row["algorithm"] for row in rows} == set(ARCHIVE_FAMILY_DEFAULT_ALGORITHMS)
    assert "archive_mode" in rows[0]
    assert "archive_size" in rows[0]
    assert "archive_subset_size" in rows[0]
    assert "archive_subset_hv" in rows[0]
    assert "hybrid_status" in rows[0]
    assert "hybrid_local_only_generations" in rows[0]

    with runs_path.open("r", encoding="utf-8", newline="") as fh:
        run_rows = list(csv.DictReader(fh))
    assert run_rows
    assert {row["algorithm"] for row in run_rows} == set(ARCHIVE_FAMILY_DEFAULT_ALGORITHMS)
    assert "archive_subset_igd_plus" in run_rows[0]

    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert set(payload["variants"]) == set(ARCHIVE_FAMILY_DEFAULT_ALGORITHMS)
    assert payload["runs"] == len(run_rows)
    assert "nsgaii_archive_hybrid" in payload["by_variant"]
