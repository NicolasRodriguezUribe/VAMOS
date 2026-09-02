from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import vamos
from vamos.experiment.optimization_result import OptimizationResult

SCRIPTS = Path(__file__).resolve().parents[2] / "experiments" / "scripts"
EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "tuning"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from ablation_runner import run_ablation  # noqa: E402
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


def test_campaign_collector_writes_derived_tidy_table(tmp_path: Path) -> None:
    study = vamos.create_study(
        vamos.StudySpec(
            problems=["zdt1"],
            algorithms=["nsgaii"],
            seeds=[17],
            pop_size=4,
            max_evaluations=4,
            labels={"variant": "baseline"},
        ),
        output=tmp_path / "study",
    ).run()
    rows = collect_campaign(study.root, campaign="tiny")
    output = tmp_path / "analysis" / "tiny.csv"

    columns = write_tidy_csv(output, rows, core_columns=CORE_COLUMNS)

    assert len(rows) == 1
    assert {"study_id", "plan_id", "run_id", "task_id", "algorithm", "problem", "seed"}.issubset(columns)
    assert rows[0]["study_id"] == study.study_id
    assert rows[0]["plan_id"] == study.plan_id
    assert rows[0]["variant"] == "baseline"
    assert rows[0]["run_manifest_sha256"]
    assert output.is_file()
    assert "nsgaii" in output.read_text(encoding="utf-8")


def test_ablation_example_executes_canonical_studies_and_retains_traceability(tmp_path: Path) -> None:
    rows = run_ablation(
        tmp_path / "ablation",
        seeds=(3,),
        max_evaluations=4,
        populations=(("baseline", 4), ("tuned", 4)),
    )

    assert [row["variant"] for row in rows] == ["baseline", "tuned"]
    assert all(row["state"] == "succeeded" for row in rows)
    assert all(row["study_id"] and row["plan_id"] for row in rows)
    assert all(row["task_id"] and row["selected_run_id"] for row in rows)
    assert all(row["run_manifest_path"] and row["run_manifest_sha256"] for row in rows)


def test_study_config_example_uses_the_public_spec_and_planner() -> None:
    config_path = Path(__file__).resolve().parents[2] / "examples" / "configs" / "study_nsgaii.json"
    spec = vamos.StudySpec(**json.loads(config_path.read_text(encoding="utf-8")))
    report = vamos.plan_study(spec)

    assert report.status == "ready"
    assert report.plan.task_count == 6
    assert report.failure_policy == "continue"


@pytest.mark.parametrize(
    "relative_path",
    [
        "notebooks/0_basic/04_advanced_configuration.ipynb",
        "notebooks/2_advanced/28_ablation_study.ipynb",
        "notebooks/2_advanced/32_ablation_planning.ipynb",
    ],
)
def test_updated_study_notebooks_compile_and_use_only_current_vocabulary(relative_path: str) -> None:
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads((root / relative_path).read_text(encoding="utf-8"))
    source = "\n\n".join(
        "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    compile(source, relative_path, "exec")
    if "ablation" in relative_path:
        assert "StudySpec" in source
        assert "plan_study" in source


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
