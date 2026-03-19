from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-untyped]
    except Exception as exc:  # pragma: no cover - optional dependency
        raise SystemExit("Missing dependency: PyYAML. Install with: pip install pyyaml") from exc
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the NSGA-II archive-family pilot campaign.")
    parser.add_argument(
        "--config",
        default="experiments/configs/archive_family_pilot_compact.yml",
        help="YAML campaign spec. Defaults to the compact pilot campaign.",
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        help="Optional suite override. Useful for running only a subset such as NSGAII_archive_family_smoke.",
    )
    parser.add_argument(
        "--benchmark-config",
        help="Optional benchmark config override for the per-suite VAMOS bench command.",
    )
    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Skip the pilot report step after running the benchmark suites.",
    )
    return parser


def _run_command(cmd: list[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fh:
        process = subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT)
    return int(process.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    repo = Path.cwd()
    spec_path = (repo / args.config).resolve()
    spec = load_yaml(spec_path)

    campaign = str(spec.get("campaign") or spec_path.stem)
    output_root = (repo / str(spec["output_root"])).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    benchmark_config = (repo / str(args.benchmark_config or spec["benchmark_config"])).resolve()
    suites = list(args.suites or spec.get("suites") or ())
    if not suites:
        raise SystemExit("No suites configured for the archive-family pilot campaign.")

    logs_root = output_root / "logs"
    records: list[dict[str, Any]] = []

    for suite in suites:
        suite_output = output_root / str(suite)
        log_path = logs_root / f"{suite}.log"
        cmd = [
            sys.executable,
            "-m",
            "vamos.experiment.cli.main",
            "bench",
            str(suite),
            "--output",
            str(suite_output),
            "--config",
            str(benchmark_config),
        ]
        print("Running:", " ".join(cmd))
        returncode = _run_command(cmd, cwd=repo, log_path=log_path)
        records.append(
            {
                "campaign": campaign,
                "suite": suite,
                "output_dir": str(suite_output.relative_to(repo)),
                "benchmark_config": str(benchmark_config.relative_to(repo)),
                "log": str(log_path.relative_to(repo)),
                "returncode": returncode,
            }
        )

    failed = [record for record in records if record["returncode"] != 0]

    analysis_cfg = dict(spec.get("analysis") or {})
    if not args.no_analysis and not failed and bool(analysis_cfg.get("enabled", True)):
        analysis_output = output_root / str(analysis_cfg.get("output_dir") or "pilot_summary")
        analysis_log = logs_root / "pilot_report.log"
        analysis_script = repo / "experiments" / "scripts" / "report_archive_family_pilot.py"
        cmd = [
            sys.executable,
            str(analysis_script),
            "--input",
            str(output_root),
            "--output",
            str(analysis_output),
        ]
        print("Running:", " ".join(cmd))
        returncode = _run_command(cmd, cwd=repo, log_path=analysis_log)
        records.append(
            {
                "campaign": campaign,
                "suite": "__pilot_report__",
                "output_dir": str(analysis_output.relative_to(repo)),
                "benchmark_config": "",
                "log": str(analysis_log.relative_to(repo)),
                "returncode": returncode,
            }
        )
        if returncode != 0:
            failed.append(records[-1])

    index_path = output_root / "runs_index.json"
    index_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print("Wrote:", index_path.relative_to(repo))
    print("Runs:", len(records))
    print("Failed:", len(failed))

    return 0 if not failed else 10


if __name__ == "__main__":
    raise SystemExit(main())
