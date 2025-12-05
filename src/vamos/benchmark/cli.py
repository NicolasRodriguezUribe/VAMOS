from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from vamos.benchmark.report import BenchmarkReport, BenchmarkReportConfig
from vamos.benchmark.runner import BenchmarkResult, run_benchmark_suite
from vamos.benchmark.suites import get_benchmark_suite


def _load_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path).expanduser()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file '{cfg_path}' not found.")
    if cfg_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError("PyYAML is required to read YAML configs.") from exc
        with cfg_path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    with cfg_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run benchmark suites and generate paper-ready reports.")
    parser.add_argument("suite", help="Benchmark suite name (see list with --list).", nargs="?")
    parser.add_argument("--list", action="store_true", help="List available benchmark suites.")
    parser.add_argument("--algorithms", nargs="+", help="Algorithms to run (default: suite defaults).")
    parser.add_argument("--metrics", nargs="+", help="Metrics to compute (default: suite defaults).")
    parser.add_argument("--output", required=False, help="Output directory for reports and runs.")
    parser.add_argument("--config", help="JSON/YAML with ExperimentConfig overrides (population size, budgets, etc.).")
    parser.add_argument("--only-report", action="store_true", help="Skip execution and regenerate reports from existing outputs.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    from vamos.benchmark.suites import list_benchmark_suites

    if args.list or not args.suite:
        print("Available benchmark suites:", ", ".join(list_benchmark_suites()))
        return

    suite = get_benchmark_suite(args.suite)
    algorithms = args.algorithms or suite.default_algorithms
    metrics = args.metrics or suite.default_metrics
    if not args.output:
        parser.error("--output is required unless --list is used.")
    base_output_dir = Path(args.output).expanduser().resolve()
    overrides = _load_config(args.config)

    if args.only_report:
        result = BenchmarkResult(
            suite=suite,
            algorithms=list(algorithms),
            metrics=list(metrics),
            base_output_dir=base_output_dir,
            summary_path=base_output_dir / "summary" / "metrics.csv",
            runs=[],
            raw_results=None,
        )
    else:
        result = run_benchmark_suite(
            suite=suite,
            algorithms=algorithms,
            metrics=metrics,
            base_output_dir=base_output_dir,
            global_config_overrides=overrides,
        )

    report_output = base_output_dir / "summary"
    report = BenchmarkReport(
        result=result,
        config=BenchmarkReportConfig(metrics=list(metrics)),
        output_dir=report_output,
    )
    tidy = report.aggregate_metrics()
    stats = report.compute_statistics()
    tables = report.generate_latex_tables()
    plots = report.generate_plots()

    print(f"[Benchmark] Suite '{suite.name}' completed.")
    print(f"[Benchmark] Summary CSV: {result.summary_path}")
    print(f"[Benchmark] Tidy metrics: {report_output / 'metrics_tidy.csv'}")
    if tables:
        print(f"[Benchmark] LaTeX tables in {report_output / 'tables'}")
    if plots:
        print(f"[Benchmark] Plots in {report_output / 'plots'}")


if __name__ == "__main__":
    main()
