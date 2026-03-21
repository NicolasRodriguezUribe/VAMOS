from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_transfer_script_smoke(tmp_path: Path) -> None:
    config_path = tmp_path / "transfer_config.json"
    output_dir = tmp_path / "transfer_out"
    config_path.write_text(
        json.dumps(
            {
                "engine": "numpy",
                "directions": [["nsgaii", "moead"]],
                "problems": ["zdt1"],
                "seeds": [0],
                "n_var": 6,
                "population_size": 8,
                "source_max_evaluations": 24,
                "target_max_evaluations": 24,
                "credit_model": "simple_improvement",
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "experiments/scripts/run_online_control_transfer.py",
            "--config",
            str(config_path),
            "--output",
            str(output_dir),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert (output_dir / "transfer_runs.csv").exists()
    assert (output_dir / "transfer_summary.csv").exists()
