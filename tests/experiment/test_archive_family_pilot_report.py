from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from vamos.experiment.benchmark.runner import run_benchmark_suite
from vamos.experiment.benchmark.suites import get_benchmark_suite


def test_archive_family_pilot_report_runs_on_smoke_benchmark(tmp_path: Path):
    suite_root = tmp_path / "campaign" / "NSGAII_archive_family_smoke"
    suite = get_benchmark_suite("NSGAII_archive_family_smoke")
    run_benchmark_suite(
        suite=suite,
        algorithms=None,
        metrics=None,
        base_output_dir=suite_root,
        global_config_overrides={
            "engine": "numpy",
            "population_size": 8,
            "offspring_population_size": 8,
        },
    )

    output_dir = tmp_path / "campaign" / "pilot_summary"
    script = Path.cwd() / "experiments" / "scripts" / "report_archive_family_pilot.py"
    subprocess.run(
        [sys.executable, str(script), "--input", str(tmp_path / "campaign"), "--output", str(output_dir)],
        cwd=Path.cwd(),
        check=True,
    )

    assert (output_dir / "archive_family_pilot_summary.md").exists()
    assert (output_dir / "archive_family_pilot_tables.csv").exists()
    assert (output_dir / "archive_family_pilot_by_family.csv").exists()
    assert (output_dir / "archive_family_pilot_by_objectives.csv").exists()
    assert (output_dir / "archive_family_pilot_diagnostics.csv").exists()
    assert (output_dir / "archive_family_pilot_regimes.csv").exists()

    content = (output_dir / "archive_family_pilot_summary.md").read_text(encoding="utf-8")
    assert "Most Promising Regimes For `hybrid_survival`" in content
    assert "Baseline has no archive subset by design" in content
