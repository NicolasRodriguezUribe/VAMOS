"""
Analyze a 2x2 MIC experiment: archive off/on x AOS off/on.

Expected variants:
  - baseline
  - aos
  - baseline_archive
  - aos_archive
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze archive x AOS factorial MIC experiment.")
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("experiments/mic/mic_factorial_archive_aos.csv"),
        help="Input experiment CSV.",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=5e-4,
        help="Tie threshold for wins/ties/losses on HV deltas.",
    )
    ap.add_argument(
        "--out-per-problem",
        type=Path,
        default=Path("experiments/mic/mic_factorial_archive_aos_per_problem.csv"),
        help="Output per-problem summary CSV.",
    )
    ap.add_argument(
        "--out-summary",
        type=Path,
        default=Path("experiments/mic/mic_factorial_archive_aos_summary.csv"),
        help="Output aggregate summary CSV.",
    )
    return ap.parse_args()


def _wtl(series: pd.Series, threshold: float) -> tuple[int, int, int]:
    wins = int((series > threshold).sum())
    losses = int((series < -threshold).sum())
    ties = int(series.shape[0] - wins - losses)
    return wins, ties, losses


def main() -> None:
    args = _parse_args()
    if not args.csv.is_file():
        raise FileNotFoundError(f"Missing CSV: {args.csv}")

    df = pd.read_csv(args.csv)
    if df.empty:
        raise ValueError(f"Empty CSV: {args.csv}")

    required = {"variant", "problem", "hypervolume", "runtime_seconds"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {args.csv}: {sorted(missing)}")

    df["variant"] = df["variant"].astype(str).str.strip().str.lower()
    df["problem"] = df["problem"].astype(str).str.strip().str.lower()

    variants = ["baseline", "aos", "baseline_archive", "aos_archive"]
    sub = df[df["variant"].isin(variants)].copy()
    if sub.empty:
        raise ValueError("No rows found for factorial variants in input CSV.")

    present = set(sub["variant"].unique())
    missing_variants = [v for v in variants if v not in present]
    if missing_variants:
        raise ValueError(f"Missing factorial variants in CSV: {missing_variants}")

    med = (
        sub.groupby(["problem", "variant"], as_index=False)
        .agg(
            hv_median=("hypervolume", "median"),
            runtime_median=("runtime_seconds", "median"),
        )
    )
    piv_hv = med.pivot(index="problem", columns="variant", values="hv_median")
    piv_rt = med.pivot(index="problem", columns="variant", values="runtime_median")

    complete = piv_hv.dropna(subset=variants).index
    if len(complete) < len(piv_hv):
        dropped = sorted(set(piv_hv.index) - set(complete))
        print(f"Warning: dropping problems with incomplete variant coverage: {dropped}")
    piv_hv = piv_hv.loc[complete].copy()
    piv_rt = piv_rt.loc[complete].copy()

    per_problem = pd.DataFrame(index=piv_hv.index)
    per_problem["hv_baseline"] = piv_hv["baseline"]
    per_problem["hv_aos"] = piv_hv["aos"]
    per_problem["hv_baseline_archive"] = piv_hv["baseline_archive"]
    per_problem["hv_aos_archive"] = piv_hv["aos_archive"]

    per_problem["rt_baseline"] = piv_rt["baseline"]
    per_problem["rt_aos"] = piv_rt["aos"]
    per_problem["rt_baseline_archive"] = piv_rt["baseline_archive"]
    per_problem["rt_aos_archive"] = piv_rt["aos_archive"]

    # Factor effects on HV
    per_problem["delta_aos_no_archive"] = per_problem["hv_aos"] - per_problem["hv_baseline"]
    per_problem["delta_aos_archive"] = per_problem["hv_aos_archive"] - per_problem["hv_baseline_archive"]
    per_problem["delta_archive_no_aos"] = per_problem["hv_baseline_archive"] - per_problem["hv_baseline"]
    per_problem["delta_archive_with_aos"] = per_problem["hv_aos_archive"] - per_problem["hv_aos"]
    per_problem["interaction_hv"] = per_problem["delta_aos_archive"] - per_problem["delta_aos_no_archive"]

    # Runtime effects (%)
    per_problem["rt_pct_aos_no_archive"] = 100.0 * (
        per_problem["rt_aos"] / per_problem["rt_baseline"] - 1.0
    )
    per_problem["rt_pct_aos_archive"] = 100.0 * (
        per_problem["rt_aos_archive"] / per_problem["rt_baseline_archive"] - 1.0
    )
    per_problem["rt_pct_archive_no_aos"] = 100.0 * (
        per_problem["rt_baseline_archive"] / per_problem["rt_baseline"] - 1.0
    )
    per_problem["rt_pct_archive_with_aos"] = 100.0 * (
        per_problem["rt_aos_archive"] / per_problem["rt_aos"] - 1.0
    )

    effect_cols = [
        "delta_aos_no_archive",
        "delta_aos_archive",
        "delta_archive_no_aos",
        "delta_archive_with_aos",
        "interaction_hv",
    ]
    summary_rows: list[dict[str, float | int | str]] = []
    print("Factorial HV summary")
    print("=" * 80)
    for col in effect_cols:
        s = per_problem[col]
        wins, ties, losses = _wtl(s, args.threshold)
        med_v = float(s.median())
        mean_v = float(s.mean())
        print(
            f"{col:<28} median={med_v:+.6f}  mean={mean_v:+.6f}  "
            f"W/T/L={wins}/{ties}/{losses}"
        )
        summary_rows.append(
            {
                "effect": col,
                "median_delta_hv": med_v,
                "mean_delta_hv": mean_v,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "threshold": float(args.threshold),
                "n_problems": int(s.shape[0]),
            }
        )

    print("\nMedian runtime deltas (%)")
    print("=" * 80)
    for col in [
        "rt_pct_aos_no_archive",
        "rt_pct_aos_archive",
        "rt_pct_archive_no_aos",
        "rt_pct_archive_with_aos",
    ]:
        s = per_problem[col]
        print(f"{col:<28} median={float(s.median()):+.2f}%  mean={float(s.mean()):+.2f}%")

    out_per_problem = args.out_per_problem
    out_per_problem.parent.mkdir(parents=True, exist_ok=True)
    per_problem.reset_index().rename(columns={"index": "problem"}).to_csv(out_per_problem, index=False)

    out_summary = args.out_summary
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(out_summary, index=False)

    print(f"\nWrote per-problem summary -> {out_per_problem}")
    print(f"Wrote aggregate summary   -> {out_summary}")


if __name__ == "__main__":
    main()

