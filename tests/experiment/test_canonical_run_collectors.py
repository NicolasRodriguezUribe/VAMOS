from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import vamos
from vamos.experiment.optimization_result import OptimizationResult

SCRIPTS = Path(__file__).resolve().parents[2] / "experiments" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from canonical_runs import discover_run_paths, load_canonical_run, run_row, write_tidy_csv  # noqa: E402
from collect_campaign_runs import CORE_COLUMNS, collect_campaign  # noqa: E402
from collect_hv_archive_metrics import pareto_nondominated_mask, scan_runs  # noqa: E402
from report_hv_archive_campaign import write_latex_table  # noqa: E402


@pytest.fixture
def canonical_run(tmp_path: Path) -> tuple[Path, OptimizationResult]:
    result = vamos.optimize("zdt1", pop_size=4, max_evaluations=4, seed=17)
    stored = vamos.save_result(result, tmp_path / "run", labels={"variant": "baseline"})
    return stored.root, result


def test_discovery_and_row_use_manifest_and_result_bundle(canonical_run: tuple[Path, OptimizationResult]) -> None:
    path, original = canonical_run

    assert discover_run_paths(path.parent) == [path]
    loaded = load_canonical_run(path)
    row = run_row(loaded, campaign="tiny")

    assert row["algorithm"] == "nsgaii"
    assert row["status"] == "succeeded"
    assert row["problem"] == "zdt1"
    assert row["engine"] == "numpy"
    assert row["seed"] == 17
    assert row["variant"] == "baseline"
    assert original.F is not None
    assert row["front_size"] == original.F.shape[0]
    np.testing.assert_array_equal(loaded.result.F, original.F)
    assert {item.name for item in path.iterdir()} == {"manifest.json", "result.npz", "environment.json"}


def test_campaign_collector_writes_derived_tidy_table(canonical_run: tuple[Path, OptimizationResult], tmp_path: Path) -> None:
    path, _ = canonical_run
    rows = collect_campaign(path.parent, campaign="tiny")
    output = tmp_path / "analysis" / "tiny.csv"

    columns = write_tidy_csv(output, rows, core_columns=CORE_COLUMNS)

    assert len(rows) == 1
    assert {"run_id", "task_id", "algorithm", "problem", "seed"}.issubset(columns)
    assert output.is_file()
    assert "nsgaii" in output.read_text(encoding="utf-8")


def test_analysis_collector_reads_objectives_from_canonical_bundle(
    canonical_run: tuple[Path, OptimizationResult],
) -> None:
    path, original = canonical_run
    rows = scan_runs(path.parent)

    assert len(rows) == 1
    assert original.F is not None
    np.testing.assert_array_equal(rows[0]["_objectives"], original.F)
    mask = pareto_nondominated_mask(np.asarray(rows[0]["_objectives"]))
    assert mask.dtype == np.bool_
    assert mask.shape == (original.F.shape[0],)


def test_report_table_uses_canonical_runtime_column(tmp_path: Path) -> None:
    table = tmp_path / "summary.tex"
    write_latex_table(
        table,
        [
            {
                "problem": "zdt1",
                "algorithm": "nsgaii",
                "engine": "numpy",
                "variant": "baseline",
                "hv_final": 0.5,
                "igd_plus": 0.25,
                "runtime_seconds": 1.5,
            }
        ],
    )

    rendered = table.read_text(encoding="utf-8")
    assert "$1.5_{0}$" in rendered
    assert "nan" not in rendered.lower()
