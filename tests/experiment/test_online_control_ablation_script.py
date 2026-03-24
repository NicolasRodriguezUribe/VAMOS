from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from vamos.experiment.online_control_analysis import write_csv_rows


def _write_suite_dir(base: Path, suite_name: str) -> Path:
    output_dir = base / suite_name
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(
        output_dir / "runs.csv",
        [
            {
                "suite": suite_name,
                "run_id": f"{suite_name}_r1",
                "host": "nsgaii",
                "problem": "zcat1" if suite_name == "zcat" else "zdt1",
                "variant": "adaptive_hierarchical_joint",
                "variant_group": "adaptive",
                "seed": 0,
                "time_ms": 100.0,
                "profile_start_step_time_ms": 1.0,
                "profile_router_time_ms": 1.0,
                "profile_policy_select_time_ms": 1.0,
                "profile_policy_update_time_ms": 1.0,
                "profile_decode_time_ms": 1.0,
                "profile_trace_time_ms": 1.0,
                "profile_variation_time_ms": 10.0,
                "profile_evaluation_time_ms": 70.0,
                "profile_survival_time_ms": 5.0,
                "profile_total_runtime_ms": 100.0,
            }
        ],
    )
    write_csv_rows(
        output_dir / "summary.csv",
        [
            {
                "suite": suite_name,
                "host": "nsgaii",
                "problem": "zcat1" if suite_name == "zcat" else "zdt1",
                "variant": "fixed_sbx",
                "variant_group": "fixed",
                "mean_hv": 0.6 if suite_name == "zcat" else 0.8,
                "mean_igd_plus": 1.0,
                "mean_time_ms": 90.0,
                "mean_average_reward": 0.0,
                "mean_average_overhead_ms": 0.0,
                "mean_family_concentration": 1.0,
                "mean_regime_concentration": 1.0,
                "mean_intent_concentration": 0.0,
                "mean_family_switches": 0.0,
                "mean_regime_switches": 0.0,
                "mean_intent_switches": 0.0,
                "mean_family_share_sbx_like": 1.0,
                "mean_family_share_de_like": 0.0,
                "mean_regime_share_repair": 0.0,
                "mean_regime_share_expand": 1.0,
                "mean_regime_share_refine": 0.0,
                "mean_intent_share_exploratory": 0.0,
                "mean_intent_share_balanced": 0.0,
                "mean_intent_share_local_refine": 0.0,
                "mean_intent_share_mutation_heavy": 0.0,
                "mean_intent_share_feasibility_biased": 0.0,
            },
            {
                "suite": suite_name,
                "host": "nsgaii",
                "problem": "zcat1" if suite_name == "zcat" else "zdt1",
                "variant": "adaptive_flat_parameter",
                "variant_group": "adaptive",
                "mean_hv": 0.69 if suite_name == "zcat" else 0.81,
                "mean_igd_plus": 0.9,
                "mean_time_ms": 99.0,
                "mean_average_reward": 0.3,
                "mean_average_overhead_ms": 1.0,
                "mean_family_concentration": 1.0,
                "mean_regime_concentration": 0.9,
                "mean_intent_concentration": 0.3,
                "mean_family_switches": 0.0,
                "mean_regime_switches": 0.0,
                "mean_intent_switches": 3.0,
                "mean_family_share_sbx_like": 1.0,
                "mean_family_share_de_like": 0.0,
                "mean_regime_share_repair": 0.0,
                "mean_regime_share_expand": 0.8,
                "mean_regime_share_refine": 0.2,
                "mean_intent_share_exploratory": 0.2,
                "mean_intent_share_balanced": 0.3,
                "mean_intent_share_local_refine": 0.4,
                "mean_intent_share_mutation_heavy": 0.1,
                "mean_intent_share_feasibility_biased": 0.0,
            },
            {
                "suite": suite_name,
                "host": "nsgaii",
                "problem": "zcat1" if suite_name == "zcat" else "zdt1",
                "variant": "adaptive_hierarchical_joint",
                "variant_group": "adaptive",
                "mean_hv": 0.72 if suite_name == "zcat" else 0.815,
                "mean_igd_plus": 0.85,
                "mean_time_ms": 101.0,
                "mean_average_reward": 0.35,
                "mean_average_overhead_ms": 1.0,
                "mean_family_concentration": 0.52,
                "mean_regime_concentration": 0.9,
                "mean_intent_concentration": 0.32,
                "mean_family_switches": 4.0,
                "mean_regime_switches": 0.0,
                "mean_intent_switches": 4.0,
                "mean_family_share_sbx_like": 0.6 if suite_name == "zcat" else 0.8,
                "mean_family_share_de_like": 0.4 if suite_name == "zcat" else 0.2,
                "mean_regime_share_repair": 0.0,
                "mean_regime_share_expand": 0.8,
                "mean_regime_share_refine": 0.2,
                "mean_intent_share_exploratory": 0.3 if suite_name == "zcat" else 0.2,
                "mean_intent_share_balanced": 0.2,
                "mean_intent_share_local_refine": 0.4,
                "mean_intent_share_mutation_heavy": 0.1,
                "mean_intent_share_feasibility_biased": 0.0,
            },
            {
                "suite": suite_name,
                "host": "nsgaii",
                "problem": "zcat1" if suite_name == "zcat" else "zdt1",
                "variant": "adaptive_hierarchical_joint_no_regime",
                "variant_group": "adaptive",
                "mean_hv": 0.715 if suite_name == "zcat" else 0.814,
                "mean_igd_plus": 0.86,
                "mean_time_ms": 100.0,
                "mean_average_reward": 0.34,
                "mean_average_overhead_ms": 1.0,
                "mean_family_concentration": 0.52,
                "mean_regime_concentration": 1.0,
                "mean_intent_concentration": 0.32,
                "mean_family_switches": 4.0,
                "mean_regime_switches": 0.0,
                "mean_intent_switches": 4.0,
                "mean_family_share_sbx_like": 0.6 if suite_name == "zcat" else 0.8,
                "mean_family_share_de_like": 0.4 if suite_name == "zcat" else 0.2,
                "mean_regime_share_repair": 0.0,
                "mean_regime_share_expand": 1.0,
                "mean_regime_share_refine": 0.0,
                "mean_intent_share_exploratory": 0.3 if suite_name == "zcat" else 0.2,
                "mean_intent_share_balanced": 0.2,
                "mean_intent_share_local_refine": 0.4,
                "mean_intent_share_mutation_heavy": 0.1,
                "mean_intent_share_feasibility_biased": 0.0,
            },
            {
                "suite": suite_name,
                "host": "nsgaii",
                "problem": "zcat1" if suite_name == "zcat" else "zdt1",
                "variant": "adaptive_hierarchical_joint_fixed_family_sbx",
                "variant_group": "adaptive",
                "mean_hv": 0.71 if suite_name == "zcat" else 0.812,
                "mean_igd_plus": 0.87,
                "mean_time_ms": 100.0,
                "mean_average_reward": 0.34,
                "mean_average_overhead_ms": 1.0,
                "mean_family_concentration": 1.0,
                "mean_regime_concentration": 0.9,
                "mean_intent_concentration": 0.31,
                "mean_family_switches": 0.0,
                "mean_regime_switches": 0.0,
                "mean_intent_switches": 4.0,
                "mean_family_share_sbx_like": 1.0,
                "mean_family_share_de_like": 0.0,
                "mean_regime_share_repair": 0.0,
                "mean_regime_share_expand": 0.8,
                "mean_regime_share_refine": 0.2,
                "mean_intent_share_exploratory": 0.25 if suite_name == "zcat" else 0.2,
                "mean_intent_share_balanced": 0.2,
                "mean_intent_share_local_refine": 0.45,
                "mean_intent_share_mutation_heavy": 0.1,
                "mean_intent_share_feasibility_biased": 0.0,
            }
        ],
    )
    write_csv_rows(
        output_dir / "trace_rows.csv",
        [
            {
                "suite": suite_name,
                "run_id": f"{suite_name}_r1",
                "host": "nsgaii",
                "problem": "zcat1" if suite_name == "zcat" else "zdt1",
                "variant": "adaptive_hierarchical_joint",
                "variant_group": "adaptive",
                "seed": 0,
                "step_index": 0,
                "generation": 0,
                "budget_progress": 0.1,
                "regime": "expand",
                "operator_family": "de_like" if suite_name == "zcat" else "sbx_like",
                "intent_prototype": "exploratory" if suite_name == "zcat" else "balanced",
                "bounded_reward": 0.3,
                "overhead_ms": 0.5,
            },
            {
                "suite": suite_name,
                "run_id": f"{suite_name}_r1",
                "host": "nsgaii",
                "problem": "zcat1" if suite_name == "zcat" else "zdt1",
                "variant": "adaptive_hierarchical_joint",
                "variant_group": "adaptive",
                "seed": 0,
                "step_index": 1,
                "generation": 1,
                "budget_progress": 0.9,
                "regime": "refine",
                "operator_family": "sbx_like",
                "intent_prototype": "local_refine",
                "bounded_reward": 0.5,
                "overhead_ms": 0.5,
            }
        ],
    )
    (output_dir / "resolved_config.json").write_text(
        json.dumps(
            {
                "hosts": ["nsgaii"],
                "problems": ["zcat1" if suite_name == "zcat" else "zdt1"],
                "variants": [
                    "fixed_sbx",
                    "adaptive_flat_parameter",
                    "adaptive_hierarchical_joint",
                    "adaptive_hierarchical_joint_no_regime",
                    "adaptive_hierarchical_joint_fixed_family_sbx",
                ],
                "seeds": [0],
                "population_size": 40,
                "max_evaluations": 800,
                "n_var": 30 if suite_name == "zcat" else 12,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return output_dir


def test_ablation_analysis_script_writes_expected_artifacts(tmp_path: Path) -> None:
    zcat_dir = _write_suite_dir(tmp_path, "zcat")
    anchor_dir = _write_suite_dir(tmp_path, "anchor")
    output_dir = tmp_path / "analysis"

    subprocess.run(
        [
            sys.executable,
            "experiments/scripts/analyze_online_control_ablation.py",
            "--zcat-dir",
            str(zcat_dir),
            "--anchor-dir",
            str(anchor_dir),
            "--output",
            str(output_dir),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert (output_dir / "ablation_policy_comparison.csv").exists()
    assert (output_dir / "source_attribution_summary.csv").exists()
    assert (output_dir / "overhead_profile_summary.csv").exists()
    assert (output_dir / "benchmark_sensitivity_summary.csv").exists()
    assert (output_dir / "ablation_analysis_report.json").exists()
    assert (output_dir / "ablation_findings_memo.md").exists()
    verdict = json.loads((output_dir / "ablation_final_verdict.json").read_text(encoding="utf-8"))
    assert "verdict" in verdict
