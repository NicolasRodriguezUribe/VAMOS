from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from vamos.experiment.benchmark.report_utils import ensure_dir, import_pandas


_INDICATOR_ALIASES = {
    "hv": "HV",
    "hypervolume": "HV",
    "igd": "IGD",
    "igd+": "IGD+",
    "igd_plus": "IGD+",
    "epsilon": "EP",
    "epsilon_additive": "EP",
    "eps": "EP",
    "epsilon_mult": "EPM",
    "eps_mult": "EPM",
    "avg_hausdorff": "AVGHD",
}


def _normalize_indicator_name(metric: str) -> str:
    key = metric.strip().lower()
    return _INDICATOR_ALIASES.get(key, metric)


def _needs_engine_suffix(df: Any) -> bool:
    if "engine" not in df.columns:
        return False
    counts = df.groupby("algorithm")["engine"].nunique()
    return bool((counts > 1).any())


def _algorithm_labels(df: Any) -> list[str]:
    use_engine = _needs_engine_suffix(df)

    def _format_row(row) -> str:
        algo = str(row.get("algorithm", "")).strip()
        engine = str(row.get("engine", "")).strip()
        if use_engine and engine and engine.lower() not in {"external", "none"}:
            return f"{algo}-{engine}"
        return algo

    return list(df.apply(_format_row, axis=1).tolist())


def build_quality_indicator_summary(raw_df: Any, metrics: Sequence[str], *, include_time: bool = True) -> Any:
    pd = import_pandas()
    df = raw_df.copy()
    df["Algorithm"] = _algorithm_labels(df)

    records: list[dict[str, object]] = []

    def _append_rows(indicator: str, col: str, *, scale: float | None = None) -> None:
        if col not in df.columns:
            return
        for _, row in df.iterrows():
            value = row.get(col)
            if pd.isna(value):
                continue
            if scale is not None:
                value = float(value) * scale
            execution_id = row.get("seed")
            if pd.isna(execution_id):
                execution_id = row.name
            records.append(
                {
                    "Algorithm": row["Algorithm"],
                    "Problem": row.get("problem"),
                    "ExecutionId": int(execution_id),
                    "IndicatorName": indicator,
                    "IndicatorValue": float(value),
                }
            )

    for metric in metrics:
        col = metric if metric in df.columns else f"indicator_{metric}"
        indicator = _normalize_indicator_name(metric)
        _append_rows(indicator, col)

    if include_time:
        _append_rows("Time", "time_ms", scale=1.0 / 1000.0)

    summary = pd.DataFrame.from_records(records)
    if not summary.empty:
        summary = summary.sort_values(["Algorithm", "Problem", "ExecutionId", "IndicatorName"])
    return summary


def write_quality_indicator_summary(
    raw_df: Any,
    metrics: Sequence[str],
    output_dir: Path,
    *,
    include_time: bool = True,
) -> Path:
    output_dir = ensure_dir(output_dir)
    summary = build_quality_indicator_summary(raw_df, metrics, include_time=include_time)
    output_path = output_dir / "QualityIndicatorSummary.csv"
    summary.to_csv(output_path, index=False)
    return output_path


__all__ = ["build_quality_indicator_summary", "write_quality_indicator_summary"]
