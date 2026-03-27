from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_confirmatory_alias_and_analysis_script_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = tmp_path / "confirmatory_config.json"
    output_dir = tmp_path / "confirmatory_out"
    config_path.write_text(
        json.dumps(
            {
                "engine": "numpy",
                "hosts": ["nsgaii", "moead"],
                "problems": [{"key": "zdt1", "suite": "anchor", "n_var": 6}],
                "seeds": [0],
                "variants": [
                    "fixed_sbx",
                    "semantic_prototype_sbx",
                    "adaptive_hierarchical_joint",
                    "adaptive_hierarchical_joint_no_regime",
                ],
                "population_size": 8,
                "max_evaluations": 24,
                "n_var": 6,
                "credit_model": "simple_improvement",
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "experiments/scripts/run_online_control_pilot.py",
            "--config",
            str(config_path),
            "--output",
            str(output_dir),
        ],
        check=True,
        cwd=repo_root,
    )
    subprocess.run(
        [
            sys.executable,
            "experiments/scripts/analyze_semantic_prototype_confirmatory.py",
            str(output_dir),
        ],
        check=True,
        cwd=repo_root,
    )
    subprocess.run(
        [
            sys.executable,
            "experiments/scripts/analyze_semantic_prototype_final_confirmatory.py",
            str(output_dir),
        ],
        check=True,
        cwd=repo_root,
    )

    assert (output_dir / "runs.csv").exists()
    assert (output_dir / "trace_rows.csv").exists()
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "confirmatory_summary.csv").exists()
    assert (output_dir / "confirmatory_report.json").exists()
    assert (output_dir / "confirmatory_findings_memo.md").exists()
    assert (output_dir / "confirmatory_final_verdict.json").exists()
    assert (output_dir / "final_confirmatory_summary.csv").exists()
    assert (output_dir / "final_confirmatory_tables.csv").exists()
    assert (output_dir / "final_confirmatory_report.json").exists()
    assert (output_dir / "final_confirmatory_findings_memo.md").exists()
    assert (output_dir / "final_confirmatory_verdict.json").exists()
