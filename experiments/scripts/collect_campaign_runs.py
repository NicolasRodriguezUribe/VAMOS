from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from canonical_runs import component_name, flatten_mapping, infer_suite_from_problem, write_tidy_csv

from vamos import load_study

CORE_COLUMNS = [
    "study_id",
    "plan_id",
    "run_id",
    "task_id",
    "attempt_id",
    "run_manifest_path",
    "run_manifest_sha256",
    "status",
    "campaign",
    "variant",
    "suite",
    "algorithm",
    "engine",
    "problem",
    "seed",
    "max_evaluations",
    "population_size",
    "runtime_seconds",
    "evaluations",
    "termination_reason",
    "retryable",
]


def collect_campaign(study_root: Path, *, campaign: str) -> list[dict[str, object]]:
    """Derive one traceable row per task from the canonical StudySummary."""
    study = load_study(study_root)
    summary = study.summarize()
    variant = study.spec.labels.get("variant")
    rows: list[dict[str, object]] = []
    for item in summary.rows:
        problem = component_name_from_id(item.problem_id)
        row: dict[str, Any] = {
            "study_id": summary.study_id,
            "plan_id": summary.plan_id,
            "run_id": item.evidence_run_id,
            "task_id": item.task_id,
            "attempt_id": item.selected_attempt_id or item.latest_attempt_id,
            "run_manifest_path": item.run_manifest_path,
            "run_manifest_sha256": item.run_manifest_sha256,
            "status": item.run_status or item.state,
            "campaign": campaign,
            "variant": variant,
            "suite": infer_suite_from_problem(problem),
            "algorithm": component_name_from_id(item.algorithm_id),
            "engine": component_name_from_id(item.backend_id),
            "problem": problem,
            "seed": item.seed,
            "max_evaluations": item.evaluation_budget,
            "population_size": item.population_size,
            "runtime_seconds": runtime_seconds(item.runtime_ms),
            "evaluations": item.evaluations,
            "termination_reason": item.termination_reason,
            "retryable": item.retryable,
        }
        if item.metrics is not None:
            flatten_mapping("metrics", item.metrics, row)
        rows.append(row)
    return rows


def component_name_from_id(component_id: str | None) -> str:
    if component_id is None:
        return "unknown"
    return component_name({"component_id": component_id})


def runtime_seconds(value: float | int | None) -> float | None:
    return float(value) / 1_000.0 if value is not None else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, help="Campaign label written to the derived table")
    parser.add_argument("--study-root", required=True, help="Canonical StudyManifest directory")
    parser.add_argument("--out", default=None, help="Derived tidy CSV path")
    parser.add_argument("--sample-out", default=None, help="Derived sample CSV path")
    parser.add_argument("--sample-n", type=int, default=12)
    args = parser.parse_args()

    repo = Path.cwd()
    study_root = (repo / args.study_root).resolve()
    if not study_root.exists():
        print("ERROR: study root not found:", study_root)
        return 2

    rows = collect_campaign(study_root, campaign=args.campaign)
    if not rows:
        print("ERROR: canonical study has no tasks.")
        return 3

    output = (repo / (args.out or f"artifacts/tidy/{args.campaign}.csv")).resolve()
    sample = (repo / (args.sample_out or f"experiments/sample_outputs/{args.campaign}_sample.csv")).resolve()
    columns = write_tidy_csv(output, rows, core_columns=CORE_COLUMNS)
    write_tidy_csv(sample, rows[: args.sample_n], core_columns=CORE_COLUMNS)

    print("Wrote:", output, "rows:", len(rows), "columns:", len(columns))
    print("Wrote derived sample:", sample, "rows:", min(len(rows), args.sample_n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
