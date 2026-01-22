"""
Summarize AOS operator switching from an exported trace CSV.

Inputs
------
Trace rows are produced by `paper/02_run_ablation_aos_racing_tuner.py` when:
  - VAMOS_ABLATION_AOS_TRACE_CSV is set (path), and
  - the run matches VAMOS_ABLATION_AOS_TRACE_VARIANTS / VAMOS_ABLATION_AOS_TRACE_PROBLEMS.

The trace CSV contains per-generation arm selections with an approximate eval counter.

Outputs
-------
Writes a long-format CSV with mean selection fractions per (variant, problem, stage, op_name).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _parse_csv_list(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for item in _parse_csv_list(raw):
        out.append(int(item))
    return out


def _stage_label(lo: int, hi: int | None) -> str:
    if lo <= 0 and hi is not None:
        return f"<= {hi}"
    if hi is None:
        return f"> {lo}"
    return f"{lo+1}-{hi}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="experiments/ablation_aos_trace.csv", help="Input AOS trace CSV.")
    ap.add_argument(
        "--checkpoints",
        type=str,
        default="10000,20000,50000",
        help="Comma-separated eval checkpoints to define stages (inclusive upper bounds).",
    )
    ap.add_argument("--variants", type=str, default="", help="Optional comma list to filter variants.")
    ap.add_argument("--problems", type=str, default="", help="Optional comma list to filter problems.")
    ap.add_argument("--out-csv", type=str, default="experiments/aos_switching_summary.csv", help="Output summary CSV.")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing trace CSV: {csv_path}")

    checkpoints = sorted(set(int(x) for x in _parse_int_list(args.checkpoints) if int(x) > 0))
    if not checkpoints:
        raise ValueError("--checkpoints must contain at least one positive integer.")

    df = pd.read_csv(csv_path)
    required = {"variant", "problem", "seed", "evals", "op_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Trace CSV missing columns: {sorted(missing)}")

    df["variant"] = df["variant"].astype(str).str.strip().str.lower()
    df["problem"] = df["problem"].astype(str).str.strip().str.lower()
    df["seed"] = df["seed"].astype(int)
    df["evals"] = df["evals"].astype(int)
    df["op_name"] = df["op_name"].astype(str).str.strip()

    variants = set(x.strip().lower() for x in _parse_csv_list(args.variants)) if args.variants else None
    problems = set(x.strip().lower() for x in _parse_csv_list(args.problems)) if args.problems else None
    if variants is not None:
        df = df[df["variant"].isin(variants)]
    if problems is not None:
        df = df[df["problem"].isin(problems)]

    # Stages: (0..c1], (c1..c2], ..., (c_last..inf)
    bounds = [0, *checkpoints]
    labels = []
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        labels.append(_stage_label(lo, hi))
    labels.append(_stage_label(checkpoints[-1], None))

    def _assign_stage(evals: int) -> str:
        for lo, hi, label in zip(bounds[:-1], bounds[1:], labels[:-1], strict=True):
            if evals <= hi:
                return label
        return labels[-1]

    df["stage"] = df["evals"].map(_assign_stage)

    pulls = (
        df.groupby(["variant", "problem", "seed", "stage", "op_name"], as_index=False)
        .size()
        .rename(columns={"size": "pulls"})
    )
    totals = pulls.groupby(["variant", "problem", "seed", "stage"], as_index=False)["pulls"].sum().rename(columns={"pulls": "total_pulls"})
    pulls = pulls.merge(totals, on=["variant", "problem", "seed", "stage"], how="left")
    pulls["fraction"] = pulls["pulls"] / pulls["total_pulls"].clip(lower=1)

    summary = (
        pulls.groupby(["variant", "problem", "stage", "op_name"], as_index=False)
        .agg(mean_fraction=("fraction", "mean"), std_fraction=("fraction", "std"), mean_pulls=("pulls", "mean"))
        .sort_values(["variant", "problem", "stage", "mean_fraction"], ascending=[True, True, True, False])
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)

    # Minimal console report (top-3 arms per stage)
    for (variant, problem, stage), grp in summary.groupby(["variant", "problem", "stage"]):
        top = grp.head(3)
        parts = [f"{row.op_name}={row.mean_fraction:.2f}" for row in top.itertuples(index=False)]
        print(f"{variant} {problem} {stage}: " + ", ".join(parts))


if __name__ == "__main__":
    main()

