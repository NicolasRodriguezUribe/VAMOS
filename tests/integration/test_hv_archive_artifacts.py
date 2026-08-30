from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import vamos
from vamos.engine.config.loader import load_experiment_spec

REPO = Path(__file__).resolve().parents[2]


def test_hv_archive_metrics_are_canonical(tmp_path: Path) -> None:
    source = REPO / "experiments" / "configs" / "hv_archive_validation_slice.yml"
    spec = load_experiment_spec(source)
    spec["defaults"]["output_root"] = str(tmp_path / "results")
    spec["defaults"]["population_size"] = 16
    spec["defaults"]["offspring_population_size"] = 16
    spec["defaults"]["max_evaluations"] = 64
    config = tmp_path / "hook-spec.json"
    config.write_text(json.dumps(spec), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "-m", "vamos.experiment.cli.main", "--config", str(config)],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout

    run_dirs = [path.parent for path in (tmp_path / "results").rglob("manifest.json")]
    assert len(run_dirs) == 1
    run = vamos.load_run(run_dirs[0], verify="all")
    assert {path.name for path in run.root.iterdir()} == {"manifest.json", "result.npz", "environment.json"}
    outcome = run.manifest["outcome"]
    hooks = outcome["metrics"]["hooks"]
    assert isinstance(hooks, Mapping)
    assert hooks["stopping"]["enabled"] is True
    assert hooks["archive"]["enabled"] is True
    assert hooks["stopping"]["trace"]
    assert hooks["archive"]["trace"]


def test_cli_genealogy_summary_is_in_canonical_metrics(tmp_path: Path) -> None:
    config = tmp_path / "genealogy.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "defaults": {
                    "output_root": str(tmp_path / "results"),
                    "engine": "numpy",
                    "algorithm": "nsgaii",
                    "problem": "zdt1",
                    "population_size": 8,
                    "offspring_population_size": 8,
                    "max_evaluations": 16,
                    "seed": 0,
                    "track_genealogy": True,
                },
            }
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, "-m", "vamos.experiment.cli.main", "--config", str(config)],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout

    run_dirs = [path.parent for path in (tmp_path / "results").rglob("manifest.json")]
    assert len(run_dirs) == 1
    run = vamos.load_run(run_dirs[0], verify="all")
    genealogy = run.manifest["outcome"]["metrics"]["genealogy"]
    assert genealogy["operator_stats"]
    assert genealogy["generation_contributions"]
