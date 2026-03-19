from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FINAL_METRICS: dict[str, str] = {
    "hv": "max",
    "indicator_igd_plus": "min",
}
SUBSET_METRICS: dict[str, str] = {
    "archive_subset_hv": "max",
    "archive_subset_igd_plus": "min",
}
OVERHEAD_METRICS: tuple[str, ...] = ("time_ms", "archive_size", "archive_subset_size")


def _load_pandas():
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - optional dependency
        raise SystemExit("Missing dependency: pandas. Install with: pip install pandas") from exc
    return pd


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _family_from_problem(problem: str) -> str:
    key = str(problem).strip().lower()
    if key.startswith("zdt"):
        return "ZDT"
    if key.startswith("dtlz"):
        return "DTLZ"
    if key.startswith("wfg"):
        return "WFG"
    if key.startswith("zcat"):
        return "ZCAT"
    return "OTHER"


def _discover_suite_roots(input_root: Path) -> list[Path]:
    if (input_root / "summary" / "archive_family_runs.csv").exists():
        return [input_root]
    roots = [path for path in sorted(input_root.iterdir()) if path.is_dir() and (path / "summary" / "archive_family_runs.csv").exists()]
    if not roots:
        raise FileNotFoundError(f"No archive-family benchmark outputs found under '{input_root}'.")
    return roots


def _load_suite_meta(suite_root: Path) -> dict[str, Any]:
    suite_json = suite_root / "summary" / "suite.json"
    if not suite_json.exists():
        return {}
    return dict(json.loads(suite_json.read_text(encoding="utf-8")))


def _collect_runs(input_root: Path):
    pd = _load_pandas()
    suite_roots = _discover_suite_roots(input_root)
    frames = []
    meta_rows = []
    for suite_root in suite_roots:
        summary_path = suite_root / "summary" / "archive_family_runs.csv"
        frame = pd.read_csv(summary_path)
        meta = _load_suite_meta(suite_root)
        suite_name = str(meta.get("suite") or suite_root.name)
        frame["suite"] = suite_name
        frame["family"] = frame["problem"].map(_family_from_problem)
        frame["n_obj"] = frame["n_obj"].astype(int)
        frame["seed"] = frame["seed"].astype(int)
        frames.append(frame)
        experiments = list(meta.get("experiments") or ())
        evaluation_budgets = [int(exp.get("evaluation_budget")) for exp in experiments if exp.get("evaluation_budget") is not None]
        seed_values = sorted({int(seed) for exp in experiments for seed in list(exp.get("seeds") or ())})
        meta_rows.append(
            {
                "suite": suite_name,
                "output_dir": str(suite_root),
                "algorithms": list(meta.get("algorithms") or ()),
                "metrics": list(meta.get("metrics") or ()),
                "seeds": seed_values,
                "seed_count": len(seed_values),
                "evaluation_budget_min": min(evaluation_budgets) if evaluation_budgets else None,
                "evaluation_budget_max": max(evaluation_budgets) if evaluation_budgets else None,
                "problems": [str(exp.get("problem")) for exp in experiments],
            }
        )
    if not frames:
        raise FileNotFoundError(f"No archive-family runs found under '{input_root}'.")
    combined = pd.concat(frames, ignore_index=True)
    return combined, meta_rows


def _aggregate_metric_table(df, *, group_cols: list[str], metrics: dict[str, str], scope: str):
    pd = _load_pandas()
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        sub = df.dropna(subset=[metric]).copy()
        if sub.empty:
            continue
        grouped = sub.groupby(group_cols, dropna=False)[metric]
        summary = grouped.agg(["count", "mean", "std", "median"]).reset_index()
        q25 = grouped.quantile(0.25).reset_index(name="q25")
        q75 = grouped.quantile(0.75).reset_index(name="q75")
        merged = summary.merge(q25, on=group_cols, how="left").merge(q75, on=group_cols, how="left")
        merged["iqr"] = merged["q75"] - merged["q25"]
        merged["scope"] = scope
        merged["metric"] = metric
        merged["direction"] = metrics[metric]
        merged = merged.rename(columns={"count": "runs"})
        rows.extend(merged.to_dict(orient="records"))
    return pd.DataFrame(rows)


def _aggregate_overhead_table(df):
    pd = _load_pandas()
    rows: list[dict[str, Any]] = []
    group_cols = ["family", "n_obj", "algorithm"]
    for metric in OVERHEAD_METRICS:
        if metric not in df.columns:
            continue
        sub = df.dropna(subset=[metric]).copy()
        if sub.empty:
            continue
        grouped = sub.groupby(group_cols, dropna=False)[metric]
        summary = grouped.agg(["count", "mean", "std", "median"]).reset_index()
        q25 = grouped.quantile(0.25).reset_index(name="q25")
        q75 = grouped.quantile(0.75).reset_index(name="q75")
        merged = summary.merge(q25, on=group_cols, how="left").merge(q75, on=group_cols, how="left")
        merged["iqr"] = merged["q75"] - merged["q25"]
        merged["scope"] = "overhead"
        merged["metric"] = metric
        merged["direction"] = "min" if metric == "time_ms" else "context"
        merged = merged.rename(columns={"count": "runs"})
        rows.extend(merged.to_dict(orient="records"))
    return pd.DataFrame(rows)


def _count_map(series) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in series:
        text = str(value).strip()
        if not text or text.lower() == "nan":
            continue
        counts[text] = counts.get(text, 0) + 1
    return counts


def _aggregate_diagnostics(df):
    pd = _load_pandas()
    hybrid = df[df["algorithm"] == "nsgaii_archive_hybrid"].copy()
    if hybrid.empty:
        return pd.DataFrame()

    hybrid["hybrid_reference_fraction"] = hybrid.apply(
        lambda row: float(row["hybrid_archive_reference_generations"])
        / max(
            1.0,
            float(row["hybrid_archive_reference_generations"]) + float(row["hybrid_local_only_generations"]),
        ),
        axis=1,
    )

    records: list[dict[str, Any]] = []
    grouped = hybrid.groupby(["family", "n_obj"], dropna=False)
    for (family, n_obj), sub in grouped:
        fallback_counts = _count_map(sub["hybrid_fallback_reason"])
        split_reason_counts = _count_map(sub["hybrid_split_front_reason"])
        records.append(
            {
                "family": family,
                "n_obj": int(n_obj),
                "runs": int(len(sub)),
                "hybrid_active_runs": int((sub["archive_survival_path"] == "hybrid").sum()),
                "hybrid_runtime_fallback_runs": int(sub["hybrid_fallback_reason"].fillna("").astype(str).str.len().gt(0).sum()),
                "hybrid_local_only_runs": int((sub["hybrid_local_only_generations"] > 0).sum()),
                "mean_hybrid_generations": float(sub["hybrid_generations"].mean()),
                "median_hybrid_generations": float(sub["hybrid_generations"].median()),
                "mean_archive_reference_generations": float(sub["hybrid_archive_reference_generations"].mean()),
                "median_archive_reference_generations": float(sub["hybrid_archive_reference_generations"].median()),
                "mean_local_only_generations": float(sub["hybrid_local_only_generations"].mean()),
                "median_local_only_generations": float(sub["hybrid_local_only_generations"].median()),
                "mean_no_split_generations": float(sub["hybrid_no_split_generations"].mean()),
                "median_no_split_generations": float(sub["hybrid_no_split_generations"].median()),
                "mean_reference_fraction": float(sub["hybrid_reference_fraction"].mean()),
                "median_reference_fraction": float(sub["hybrid_reference_fraction"].median()),
                "fallback_reason_counts_json": json.dumps(fallback_counts, sort_keys=True),
                "split_front_reason_counts_json": json.dumps(split_reason_counts, sort_keys=True),
            }
        )
    return pd.DataFrame(records)


def _signed_effect(candidate: float, reference: float, direction: str) -> tuple[float, float]:
    if direction == "max":
        raw = candidate - reference
    else:
        raw = reference - candidate
    scale = max(abs(candidate), abs(reference), 1e-12)
    return raw, raw / scale


def _build_comparison_rows(summary_df, *, comparisons: list[tuple[str, str, str]], scope: str):
    pd = _load_pandas()
    rows: list[dict[str, Any]] = []
    scoped = summary_df[summary_df["scope"] == scope]
    if scoped.empty:
        return pd.DataFrame()
    for metric in sorted(scoped["metric"].unique()):
        sub = scoped[scoped["metric"] == metric]
        pivot_mean = sub.pivot_table(index=["family", "n_obj"], columns="algorithm", values="mean")
        pivot_runs = sub.pivot_table(index=["family", "n_obj"], columns="algorithm", values="runs")
        direction = str(sub["direction"].iloc[0])
        for comparison_name, candidate, reference in comparisons:
            if candidate not in pivot_mean.columns or reference not in pivot_mean.columns:
                continue
            for (family, n_obj), candidate_value in pivot_mean[candidate].dropna().items():
                reference_value = pivot_mean.at[(family, n_obj), reference] if (family, n_obj) in pivot_mean.index else None
                if _safe_float(reference_value) is None:
                    continue
                raw_delta, effect = _signed_effect(float(candidate_value), float(reference_value), direction)
                rows.append(
                    {
                        "family": family,
                        "n_obj": int(n_obj),
                        "scope": scope,
                        "metric": metric,
                        "comparison": comparison_name,
                        "candidate": candidate,
                        "reference": reference,
                        "candidate_mean": float(candidate_value),
                        "reference_mean": float(reference_value),
                        "signed_delta": raw_delta,
                        "signed_effect": effect,
                        "candidate_runs": int(pivot_runs.at[(family, n_obj), candidate]),
                        "reference_runs": int(pivot_runs.at[(family, n_obj), reference]),
                    }
                )
    return pd.DataFrame(rows)


def _aggregate_comparison_dimension(comparison_df, dimension: str):
    pd = _load_pandas()
    if comparison_df.empty:
        return pd.DataFrame()
    grouped = comparison_df.groupby([dimension, "scope", "comparison"], dropna=False)
    rows = []
    for keys, sub in grouped:
        dim_value, scope, comparison = keys
        rows.append(
            {
                dimension: dim_value,
                "scope": scope,
                "comparison": comparison,
                "regimes": int(len(sub)),
                "metrics": ",".join(sorted(sub["metric"].astype(str).unique())),
                "mean_signed_effect": float(sub["signed_effect"].mean()),
                "median_signed_effect": float(sub["signed_effect"].median()),
                "mean_signed_delta": float(sub["signed_delta"].mean()),
                "positive_regimes": int((sub["signed_effect"] > 0).sum()),
                "negative_regimes": int((sub["signed_effect"] < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def _build_regime_scores(comparison_df, diagnostics_df):
    pd = _load_pandas()
    if comparison_df.empty:
        return pd.DataFrame()
    score_rows = []
    grouped = comparison_df.groupby(["family", "n_obj"], dropna=False)
    diagnostics_lookup = diagnostics_df.set_index(["family", "n_obj"]) if not diagnostics_df.empty else None

    def _comparison_signal(group, scope: str, comparison: str) -> float | None:
        sub = group[(group["scope"] == scope) & (group["comparison"] == comparison)]
        if sub.empty:
            return None
        return float(sub["signed_effect"].mean())

    for (family, n_obj), sub in grouped:
        passive_signal = _comparison_signal(sub, "final_population", "passive_vs_off")
        hybrid_signal = _comparison_signal(sub, "final_population", "hybrid_vs_passive")
        hybrid_off_signal = _comparison_signal(sub, "final_population", "hybrid_vs_off")
        subset_signal = _comparison_signal(sub, "archive_subset", "hybrid_vs_passive")
        reference_fraction = None
        local_only_generations = None
        reference_generations = None
        if diagnostics_lookup is not None and (family, n_obj) in diagnostics_lookup.index:
            diag_row = diagnostics_lookup.loc[(family, n_obj)]
            reference_fraction = float(diag_row["mean_reference_fraction"])
            local_only_generations = float(diag_row["mean_local_only_generations"])
            reference_generations = float(diag_row["mean_archive_reference_generations"])

        classification = "mixed/needs more budget"
        if reference_fraction is not None and reference_fraction < 0.35:
            classification = "hybrid mostly local-only"
        elif passive_signal is not None and passive_signal > 0.02 and abs(hybrid_signal or 0.0) <= 0.01:
            classification = "passive captures most visible gain"
        elif hybrid_signal is not None and hybrid_signal > 0.02 and (reference_fraction is None or reference_fraction >= 0.50):
            classification = "most promising for hybrid_survival"
        elif hybrid_signal is not None and hybrid_signal < -0.02:
            classification = "hybrid not yet promising"

        score_rows.append(
            {
                "family": family,
                "n_obj": int(n_obj),
                "passive_vs_off_final_signal": passive_signal,
                "hybrid_vs_passive_final_signal": hybrid_signal,
                "hybrid_vs_off_final_signal": hybrid_off_signal,
                "hybrid_vs_passive_subset_signal": subset_signal,
                "mean_reference_fraction": reference_fraction,
                "mean_archive_reference_generations": reference_generations,
                "mean_local_only_generations": local_only_generations,
                "classification": classification,
            }
        )
    return pd.DataFrame(score_rows)


def _render_markdown_table(df, columns: list[str], *, float_cols: set[str]) -> list[str]:
    if df.empty:
        return ["_No rows available._"]
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, rule]
    for _, row in df.iterrows():
        cells = []
        for column in columns:
            value = row.get(column)
            if column in float_cols and _safe_float(value) is not None:
                cells.append(f"{float(value):.4f}")
            else:
                cells.append("" if value is None else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_archive_family_pilot_report(input_root: Path, output_dir: Path) -> dict[str, Path]:
    pd = _load_pandas()
    runs_df, meta_rows = _collect_runs(input_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_table = _aggregate_metric_table(
        runs_df,
        group_cols=["family", "n_obj", "algorithm"],
        metrics=FINAL_METRICS,
        scope="final_population",
    )
    subset_table = _aggregate_metric_table(
        runs_df[runs_df["algorithm"].isin(["nsgaii_archive_passive", "nsgaii_archive_hybrid"])],
        group_cols=["family", "n_obj", "algorithm"],
        metrics=SUBSET_METRICS,
        scope="archive_subset",
    )
    overhead_table = _aggregate_overhead_table(runs_df)
    tables_df = pd.concat([final_table, subset_table, overhead_table], ignore_index=True)

    final_comparisons = _build_comparison_rows(
        final_table,
        comparisons=[
            ("passive_vs_off", "nsgaii_archive_passive", "nsgaii_archive_off"),
            ("hybrid_vs_passive", "nsgaii_archive_hybrid", "nsgaii_archive_passive"),
            ("hybrid_vs_off", "nsgaii_archive_hybrid", "nsgaii_archive_off"),
        ],
        scope="final_population",
    )
    subset_comparisons = _build_comparison_rows(
        subset_table,
        comparisons=[("hybrid_vs_passive", "nsgaii_archive_hybrid", "nsgaii_archive_passive")],
        scope="archive_subset",
    )
    comparison_df = pd.concat([final_comparisons, subset_comparisons], ignore_index=True)

    diagnostics_df = _aggregate_diagnostics(runs_df)
    family_df = _aggregate_comparison_dimension(comparison_df, "family")
    objectives_df = _aggregate_comparison_dimension(comparison_df, "n_obj")
    regimes_df = _build_regime_scores(comparison_df, diagnostics_df)

    tables_path = output_dir / "archive_family_pilot_tables.csv"
    by_family_path = output_dir / "archive_family_pilot_by_family.csv"
    by_objectives_path = output_dir / "archive_family_pilot_by_objectives.csv"
    overhead_path = output_dir / "archive_family_pilot_overhead.csv"
    diagnostics_path = output_dir / "archive_family_pilot_diagnostics.csv"
    regimes_path = output_dir / "archive_family_pilot_regimes.csv"

    tables_df.to_csv(tables_path, index=False)
    family_df.to_csv(by_family_path, index=False)
    objectives_df.to_csv(by_objectives_path, index=False)
    overhead_table.to_csv(overhead_path, index=False)
    diagnostics_df.to_csv(diagnostics_path, index=False)
    regimes_df.to_csv(regimes_path, index=False)

    suite_names = [row["suite"] for row in meta_rows]
    seed_counts = sorted({int(row["seed_count"]) for row in meta_rows if row.get("seed_count") is not None})
    family_budget_lines = []
    for row in meta_rows:
        family = row["suite"].replace("NSGAII_archive_family_", "")
        family_budget_lines.append(
            f"- `{family}`: planned evaluations {row['evaluation_budget_min']} to {row['evaluation_budget_max']}; "
            f"seeds={row['seed_count']}"
        )

    top_promising = regimes_df[
        regimes_df["classification"] == "most promising for hybrid_survival"
    ].sort_values(
        by=["hybrid_vs_passive_final_signal", "hybrid_vs_passive_subset_signal"],
        ascending=[False, False],
        na_position="last",
    ).head(5)
    passive_sufficient = regimes_df[
        regimes_df["classification"] == "passive captures most visible gain"
    ].sort_values(by=["passive_vs_off_final_signal"], ascending=False, na_position="last").head(5)
    local_only = regimes_df[
        regimes_df["classification"] == "hybrid mostly local-only"
    ].sort_values(by=["mean_reference_fraction"], ascending=True, na_position="last").head(5)

    score_table = regimes_df.sort_values(["family", "n_obj"]).copy()
    markdown_lines = [
        "# Archive-Family Pilot Summary",
        "",
        "## Study Description",
        "",
        f"- Input root: `{input_root}`",
        f"- Suites used: {', '.join(suite_names)}",
        f"- Variants compared: `nsgaii_archive_off`, `nsgaii_archive_passive`, `nsgaii_archive_hybrid`",
        f"- Unique seed-counts detected across suites: {', '.join(str(count) for count in seed_counts) if seed_counts else 'unknown'}",
        "- Final-population metrics: `hv`, `igd_plus` when available",
        "- Archive-subset metrics: `archive_subset_hv`, `archive_subset_igd_plus` for passive and hybrid only",
        "- Baseline has no archive subset by design, so subset comparisons are passive vs hybrid only",
        "",
        "Planned evaluation budgets by suite:",
        *family_budget_lines,
        "",
        "## Final-Population Signals By Family/Objectives",
        "",
        "Heuristic signal columns are signed mean effects over the available final-population metrics. Positive means the left-hand variant looked better on average for that regime; negative means worse. These are pilot-effect summaries, not significance claims.",
        "",
        *_render_markdown_table(
            score_table[[
                "family",
                "n_obj",
                "passive_vs_off_final_signal",
                "hybrid_vs_passive_final_signal",
                "hybrid_vs_off_final_signal",
                "classification",
            ]],
            [
                "family",
                "n_obj",
                "passive_vs_off_final_signal",
                "hybrid_vs_passive_final_signal",
                "hybrid_vs_off_final_signal",
                "classification",
            ],
            float_cols={
                "passive_vs_off_final_signal",
                "hybrid_vs_passive_final_signal",
                "hybrid_vs_off_final_signal",
            },
        ),
        "",
        "## Archive-Subset Signals",
        "",
        "Subset signals compare `hybrid_survival` against `passive` on the exported archive subset only.",
        "",
        *_render_markdown_table(
            score_table[[
                "family",
                "n_obj",
                "hybrid_vs_passive_subset_signal",
                "mean_reference_fraction",
            ]],
            ["family", "n_obj", "hybrid_vs_passive_subset_signal", "mean_reference_fraction"],
            float_cols={"hybrid_vs_passive_subset_signal", "mean_reference_fraction"},
        ),
        "",
        "## Hybrid Diagnostics",
        "",
        *_render_markdown_table(
            diagnostics_df.sort_values(["family", "n_obj"])[[
                "family",
                "n_obj",
                "hybrid_active_runs",
                "hybrid_runtime_fallback_runs",
                "mean_archive_reference_generations",
                "mean_local_only_generations",
                "mean_reference_fraction",
            ]],
            [
                "family",
                "n_obj",
                "hybrid_active_runs",
                "hybrid_runtime_fallback_runs",
                "mean_archive_reference_generations",
                "mean_local_only_generations",
                "mean_reference_fraction",
            ],
            float_cols={
                "mean_archive_reference_generations",
                "mean_local_only_generations",
                "mean_reference_fraction",
            },
        ),
        "",
        "## Overhead Snapshot",
        "",
        *_render_markdown_table(
            overhead_table[
                overhead_table["metric"].isin(["time_ms", "archive_size", "archive_subset_size"])
            ].sort_values(["family", "n_obj", "algorithm", "metric"])[
                ["family", "n_obj", "algorithm", "metric", "mean", "median"]
            ].head(18),
            ["family", "n_obj", "algorithm", "metric", "mean", "median"],
            float_cols={"mean", "median"},
        ),
        "",
        "## Heuristic Decision Summary",
        "",
        "These bullets are meant to prioritize the next larger campaign, not to make final publication claims.",
        "",
        "### Most Promising Regimes For `hybrid_survival`",
        "",
        *_render_markdown_table(
            top_promising[[
                "family",
                "n_obj",
                "hybrid_vs_passive_final_signal",
                "hybrid_vs_passive_subset_signal",
                "mean_reference_fraction",
            ]],
            [
                "family",
                "n_obj",
                "hybrid_vs_passive_final_signal",
                "hybrid_vs_passive_subset_signal",
                "mean_reference_fraction",
            ],
            float_cols={
                "hybrid_vs_passive_final_signal",
                "hybrid_vs_passive_subset_signal",
                "mean_reference_fraction",
            },
        ),
        "",
        "### Regimes Where Passive Already Captures Most Of The Visible Gain",
        "",
        *_render_markdown_table(
            passive_sufficient[[
                "family",
                "n_obj",
                "passive_vs_off_final_signal",
                "hybrid_vs_passive_final_signal",
                "classification",
            ]],
            [
                "family",
                "n_obj",
                "passive_vs_off_final_signal",
                "hybrid_vs_passive_final_signal",
                "classification",
            ],
            float_cols={"passive_vs_off_final_signal", "hybrid_vs_passive_final_signal"},
        ),
        "",
        "### Regimes Where Hybrid Was Mostly Local-Only",
        "",
        *_render_markdown_table(
            local_only[[
                "family",
                "n_obj",
                "mean_reference_fraction",
                "mean_archive_reference_generations",
                "mean_local_only_generations",
            ]],
            [
                "family",
                "n_obj",
                "mean_reference_fraction",
                "mean_archive_reference_generations",
                "mean_local_only_generations",
            ],
            float_cols={
                "mean_reference_fraction",
                "mean_archive_reference_generations",
                "mean_local_only_generations",
            },
        ),
        "",
        "## Interpretation Notes",
        "",
        "- Positive final-population signals for `passive_vs_off` suggest that maintaining the archive may already help in that regime.",
        "- Positive `hybrid_vs_passive` signals suggest that archive-aware split-front survival may be adding value beyond passive archival alone.",
        "- Low `mean_reference_fraction` means hybrid had limited opportunity to use historical archive novelty and spent more time behaving like local-only split-front selection.",
        "- Because the baseline has no archive subset, compare subset metrics only between passive and hybrid, and use final-population metrics for baseline vs archive-family comparisons.",
        "",
        "Related CSV artifacts:",
        f"- `{tables_path.name}`: per-family/per-objective aggregate tables for final population, archive subset, and overhead.",
        f"- `{by_family_path.name}`: family-level comparison summaries.",
        f"- `{by_objectives_path.name}`: objective-count comparison summaries.",
        f"- `{diagnostics_path.name}`: hybrid activation and fallback diagnostics.",
        f"- `{regimes_path.name}`: heuristic regime ranking and classification.",
    ]

    markdown_path = output_dir / "archive_family_pilot_summary.md"
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    return {
        "tables": tables_path,
        "by_family": by_family_path,
        "by_objectives": by_objectives_path,
        "overhead": overhead_path,
        "diagnostics": diagnostics_path,
        "regimes": regimes_path,
        "summary": markdown_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a compact pilot-study summary from archive-family benchmark outputs.")
    parser.add_argument("--input", required=True, help="Benchmark campaign root or a single benchmark suite output directory.")
    parser.add_argument(
        "--output",
        help="Output directory for pilot artifacts. Defaults to <input>/pilot_summary when analyzing a campaign root.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    input_root = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve() if args.output else input_root / "pilot_summary"
    outputs = write_archive_family_pilot_report(input_root, output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
