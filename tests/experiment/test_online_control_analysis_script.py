from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from vamos.experiment.online_control_analysis import write_csv_rows


def test_analysis_script_writes_expected_artifacts(tmp_path: Path) -> None:
    write_csv_rows(
        tmp_path / "runs.csv",
        [
            {
                "run_id": "r1",
                "host": "nsgaii",
                "problem": "zdt1",
                "variant": "fixed_sbx",
                "variant_group": "fixed",
                "seed": 0,
                "time_ms": 10.0,
            }
        ],
    )
    write_csv_rows(
        tmp_path / "summary.csv",
        [
            {
                "host": "nsgaii",
                "problem": "zdt1",
                "variant": "fixed_sbx",
                "variant_group": "fixed",
                "mean_hv": 1.0,
                "mean_igd_plus": 1.0,
                "mean_time_ms": 10.0,
                "mean_average_reward": 0.0,
                "mean_average_overhead_ms": 0.0,
                "mean_family_concentration": 1.0,
                "mean_regime_concentration": 0.0,
                "mean_intent_concentration": 0.0,
                "mean_family_switches": 0.0,
                "mean_regime_switches": 0.0,
                "mean_intent_switches": 0.0,
                "mean_family_share_sbx_like": 1.0,
                "mean_family_share_de_like": 0.0,
                "mean_regime_share_repair": 0.0,
                "mean_regime_share_expand": 0.0,
                "mean_regime_share_refine": 0.0,
                "mean_intent_share_exploratory": 0.0,
                "mean_intent_share_balanced": 0.0,
                "mean_intent_share_local_refine": 0.0,
                "mean_intent_share_mutation_heavy": 0.0,
                "mean_intent_share_feasibility_biased": 0.0,
            }
        ],
    )
    write_csv_rows(tmp_path / "trace_rows.csv", [])

    subprocess.run(
        [sys.executable, "experiments/scripts/analyze_online_control_pilot.py", str(tmp_path)],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert (tmp_path / "policy_comparison.csv").exists()
    assert (tmp_path / "problem_host_summary.csv").exists()
    assert (tmp_path / "concentration_summary.csv").exists()
    assert (tmp_path / "heterogeneity_summary.csv").exists()
    assert (tmp_path / "phase_summary.csv").exists()
    report = json.loads((tmp_path / "analysis_report.json").read_text(encoding="utf-8"))
    assert "counts" in report
