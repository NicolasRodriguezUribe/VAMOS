from __future__ import annotations

from pathlib import Path

CALLERS = (
    "src/vamos/experiment/ablation.py",
    "src/vamos/experiment/study_analysis.py",
    "src/vamos/experiment/benchmark/cli.py",
    "src/vamos/experiment/benchmark/runner.py",
    "src/vamos/experiment/runner.py",
    "src/vamos/experiment/cli/ablation.py",
    "src/vamos/ux/studio/data.py",
    "src/vamos/ux/studio/services.py",
    "src/vamos/ux/analysis/tuning_viz.py",
)


def test_active_package_study_callers_do_not_import_superseded_runtime() -> None:
    root = Path(__file__).resolve().parents[2]
    forbidden = (
        "experiment.study.api import",
        "experiment.study.runner import",
        "experiment.study.types import",
        "StudyRunner",
        "StudyTask",
        "study_results_to_dataframe",
    )
    for relative in CALLERS:
        source = (root / relative).read_text(encoding="utf-8")
        assert not any(token in source for token in forbidden), relative


def test_studio_uses_canonical_study_traversal_not_run_manifest_discovery() -> None:
    root = Path(__file__).resolve().parents[2]
    data_source = (root / "src/vamos/ux/studio/data.py").read_text(encoding="utf-8")
    services_source = (root / "src/vamos/ux/studio/services.py").read_text(encoding="utf-8")
    assert 'rglob("manifest.json")' not in data_source
    assert 'rglob("manifest.json")' not in services_source
    assert "load_study(" in data_source
    assert "study.summarize()" in data_source
