from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_pilot_script_smoke_covers_fixed_de_on_both_hosts(tmp_path: Path) -> None:
    config_path = tmp_path / "pilot_config.json"
    output_dir = tmp_path / "pilot_out"
    config_path.write_text(
        json.dumps(
            {
                "engine": "numpy",
                "hosts": ["nsgaii", "moead"],
                "problems": ["zdt1"],
                "seeds": [0],
                "variants": ["fixed_sbx", "fixed_de", "adaptive_hierarchical_joint"],
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
        cwd=Path(__file__).resolve().parents[2],
    )

    assert (output_dir / "runs.csv").exists()
    assert (output_dir / "trace_rows.csv").exists()
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "go_no_go_summary.json").exists()
