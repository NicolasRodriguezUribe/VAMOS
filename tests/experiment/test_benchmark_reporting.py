import json
from pathlib import Path

import numpy as np
import pytest

from vamos.experiment.benchmark.suites import get_benchmark_suite, list_benchmark_suites
from vamos.experiment.benchmark.runner import BenchmarkResult
from vamos.experiment.benchmark.report import BenchmarkReport, BenchmarkReportConfig


def test_suite_registry_contains_defaults():
    names = list_benchmark_suites()
    assert "ZDT_small" in names
    assert "CEC2009_UF_CF_curved" in names
    assert "LSMOP_large" in names
    assert "Constrained_CDTLZ_MW_DCDTLZ" in names
    suite = get_benchmark_suite("ZDT_small")
    assert suite.experiments
    assert suite.default_algorithms


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_report_pipeline_on_fake_csv(tmp_path: Path):
    pd = pytest.importorskip("pandas")
    suite = get_benchmark_suite("ZDT_small")
    summary_dir = tmp_path / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for problem in ["zdt1", "zdt2"]:
        for alg in ["a1", "a2"]:
            for seed in [0, 1]:
                rows.append(
                    {
                        "problem": problem,
                        "algorithm": alg,
                        "engine": "numpy",
                        "seed": seed,
                        "n_var": 30,
                        "n_obj": 2,
                        "hv": 0.8 + 0.01 * np.random.rand(),
                        "indicator_igd_plus": 0.2 + 0.01 * np.random.rand(),
                    }
                )
    df = pd.DataFrame(rows)
    summary_path = summary_dir / "metrics.csv"
    df.to_csv(summary_path, index=False)

    # Minimal metadata
    meta = {"suite": suite.name, "algorithms": ["a1", "a2"], "metrics": ["hv", "igd_plus"]}
    (summary_dir / "suite.json").write_text(json.dumps(meta), encoding="utf-8")

    result = BenchmarkResult(
        suite=suite,
        algorithms=["a1", "a2"],
        metrics=["hv", "igd_plus"],
        base_output_dir=tmp_path,
        summary_path=summary_path,
        runs=[],
        raw_results=None,
    )
    report = BenchmarkReport(
        result=result,
        config=BenchmarkReportConfig(metrics=["hv", "igd_plus"], alpha=0.1),
        output_dir=summary_dir,
    )

    tidy = report.aggregate_metrics()
    assert not tidy.empty
    assert set(tidy["metric"].unique()) == {"hv", "igd_plus"}

    stats = report.compute_statistics()
    assert "hv" in stats

    tables = report.generate_latex_tables()
    hv_table = tables.get("hv")
    assert hv_table is not None and hv_table.exists()
    content = hv_table.read_text(encoding="utf-8")
    assert "\\textbf" in content

    # Plots are optional; ensure the function does not raise.
    report.generate_plots()
