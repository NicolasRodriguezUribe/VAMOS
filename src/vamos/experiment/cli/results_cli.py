from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from vamos.ux.analysis.results import discover_runs


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _format_timestamp(value: object) -> str:
    parsed = _parse_timestamp(value)
    if not parsed:
        return "-"
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def _load_records(base_dir: str, *, problem: str | None, algorithm: str | None, engine: str | None) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for run in discover_runs(base_dir):
        if problem and run.problem != problem:
            continue
        if algorithm and run.algorithm != algorithm:
            continue
        if engine and run.engine != engine:
            continue
        try:
            metadata = json.loads(run.metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        metrics = metadata.get("metrics", {}) or {}
        record = {
            "timestamp": metadata.get("timestamp"),
            "problem": metadata.get("problem", {}).get("key", run.problem),
            "algorithm": metadata.get("algorithm", run.algorithm),
            "engine": metadata.get("backend", run.engine),
            "seed": metadata.get("seed", run.seed),
            "evaluations": metrics.get("evaluations"),
            "time_ms": metrics.get("time_ms"),
            "evals_per_sec": metrics.get("evals_per_sec"),
            "termination": metrics.get("termination"),
            "path": str(run.path),
        }
        record["_timestamp"] = _parse_timestamp(record["timestamp"])
        records.append(record)
    records.sort(key=_record_sort_key, reverse=True)
    return records


def _record_sort_key(row: dict[str, object]) -> datetime:
    value = row.get("_timestamp")
    if isinstance(value, datetime):
        return value
    return datetime.min


def _render_table(rows: list[dict[str, object]], *, show_paths: bool) -> str:
    headers = ["timestamp", "problem", "algorithm", "engine", "seed", "evaluations", "time_ms", "evals_per_sec", "termination"]
    if show_paths:
        headers.append("path")
    formatted: list[dict[str, str]] = []
    for row in rows:
        formatted_row: dict[str, str] = {
            "timestamp": _format_timestamp(row.get("timestamp")),
            "problem": str(row.get("problem", "")),
            "algorithm": str(row.get("algorithm", "")),
            "engine": str(row.get("engine", "")),
            "seed": "-" if row.get("seed") is None else str(row.get("seed")),
            "evaluations": "-" if row.get("evaluations") is None else str(row.get("evaluations")),
            "time_ms": "-" if row.get("time_ms") is None else f"{row.get('time_ms'):.2f}",
            "evals_per_sec": "-" if row.get("evals_per_sec") is None else f"{row.get('evals_per_sec'):.1f}",
            "termination": str(row.get("termination") or "-"),
        }
        if show_paths:
            formatted_row["path"] = str(row.get("path", ""))
        formatted.append(formatted_row)
    widths = {key: max(len(key), *(len(row[key]) for row in formatted)) for key in headers}
    header_line = "  ".join(key.upper().ljust(widths[key]) for key in headers)
    sep_line = "  ".join("-" * widths[key] for key in headers)
    lines = [header_line, sep_line]
    for formatted_row in formatted:
        lines.append("  ".join(formatted_row[key].ljust(widths[key]) for key in headers))
    return "\n".join(lines)


def run_summarize(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="vamos summarize",
        description="Summarize completed runs under a results directory.",
    )
    parser.add_argument("--results", default="results", help="Results root directory to scan (default: results).")
    parser.add_argument("--limit", type=int, default=20, help="Max number of runs to display (default: 20).")
    parser.add_argument("--latest", action="store_true", help="Show only the most recent run.")
    parser.add_argument("--problem", help="Filter by problem key (e.g., zdt1).")
    parser.add_argument("--algorithm", help="Filter by algorithm name (e.g., nsgaii).")
    parser.add_argument("--engine", help="Filter by engine/backend (e.g., numpy).")
    parser.add_argument("--format", choices=("table", "json"), default="table", help="Output format (default: table).")
    parser.add_argument("--show-paths", action="store_true", help="Include run output paths in the table.")
    args = parser.parse_args(argv)

    records = _load_records(args.results, problem=args.problem, algorithm=args.algorithm, engine=args.engine)
    if not records:
        print(f"No runs found under {args.results}.")
        return
    if args.latest:
        records = records[:1]
    elif args.limit is not None and args.limit > 0:
        records = records[: args.limit]

    if args.format == "json":
        payload = [{k: v for k, v in row.items() if not k.startswith("_")} for row in records]
        print(json.dumps(payload, indent=2))
        return

    try:
        import pandas as pd
    except Exception:
        pd = None
    if pd is not None:
        payload = [{k: v for k, v in row.items() if not k.startswith("_")} for row in records]
        df = pd.DataFrame(payload)
        if not args.show_paths and "path" in df.columns:
            df = df.drop(columns=["path"])
        if "timestamp" in df.columns:
            df["timestamp"] = df["timestamp"].map(_format_timestamp)
        print(df.to_string(index=False))
        return

    table = _render_table(records, show_paths=args.show_paths)
    print(table)


def _open_path(path: Path) -> bool:
    try:
        if hasattr(os, "startfile"):
            os.startfile(path)
            return True
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
            return True
        subprocess.run(["xdg-open", str(path)], check=False)
        return True
    except Exception:
        return False


def run_open_results(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="vamos open-results",
        description="Locate the latest run directory and optionally open it.",
    )
    parser.add_argument("--results", default="results", help="Results root directory to scan (default: results).")
    parser.add_argument("--problem", help="Filter by problem key (e.g., zdt1).")
    parser.add_argument("--algorithm", help="Filter by algorithm name (e.g., nsgaii).")
    parser.add_argument("--engine", help="Filter by engine/backend (e.g., numpy).")
    parser.add_argument("--open", action="store_true", help="Open the latest run folder in your file explorer.")
    args = parser.parse_args(argv)

    records = _load_records(args.results, problem=args.problem, algorithm=args.algorithm, engine=args.engine)
    if not records:
        print(f"No runs found under {args.results}.")
        return
    latest = records[0]
    path = Path(str(latest.get("path")))
    print(f"Latest run: {path}")
    if not args.open:
        print("Tip: use --open to open this folder in your file explorer.")
        return
    opened = _open_path(path)
    if not opened:
        print(f"Unable to open {path}. Please open it manually.")


__all__ = ["run_summarize", "run_open_results"]
