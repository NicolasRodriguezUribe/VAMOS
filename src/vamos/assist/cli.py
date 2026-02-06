from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .apply import apply_plan
from .catalog import build_catalog
from .doctor import collect_doctor_report, format_doctor_report_text
from .explain import summarize_plan
from .go import go
from .plan import create_plan
from .providers import MockPlanProvider, OpenAIPlanProvider, PlanProvider
from .run import run_plan


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vamos assist",
        description="Assistant utilities for metadata discovery.",
    )
    subcommands = parser.add_subparsers(dest="assist_command", required=True)

    catalog_parser = subcommands.add_parser("catalog", help="Show available algorithms, kernels, operators, and templates.")
    catalog_parser.add_argument(
        "--problem-type",
        choices=("real", "int", "binary"),
        default="real",
        help="Operator problem type used for crossover/mutation listing (default: real).",
    )
    catalog_parser.add_argument("--json", action="store_true", help="Print JSON output.")

    plan_parser = subcommands.add_parser("plan", help="Create a template-first assist plan directory.")
    plan_parser.add_argument("prompt", help="Prompt text describing intent.")
    plan_parser.add_argument("--template", help="Quickstart template key (optional in interactive terminals).")
    plan_parser.add_argument(
        "--mode",
        choices=("template", "auto"),
        default="template",
        help="Planning mode (default: template).",
    )
    plan_parser.add_argument(
        "--provider",
        choices=("mock", "openai"),
        help=(
            "Provider name for --mode auto: mock|openai. "
            'OpenAI setup: pip install vamos[openai] (or pip install openai); setx OPENAI_API_KEY "..."; '
            'python -m vamos.experiment.cli.main assist plan "..." --mode auto --provider openai --json'
        ),
    )
    plan_parser.add_argument("--out", help="Output plan directory path.")
    plan_parser.add_argument(
        "--problem-type",
        choices=("real", "int", "binary"),
        default="real",
        help="Operator problem type used for catalog generation (default: real).",
    )
    plan_parser.add_argument("--json", action="store_true", help="Print JSON output.")

    apply_parser = subcommands.add_parser("apply", help="Materialize a runnable project directory from an assist plan.")
    apply_parser.add_argument("--plan", required=True, help="Path to an existing plan directory.")
    apply_parser.add_argument("--out", help="Output project directory path.")
    apply_parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if it already exists.")
    apply_parser.add_argument("--json", action="store_true", help="Print JSON output.")

    run_parser = subcommands.add_parser("run", help="Execute an assist plan/project config and write run metadata.")
    run_parser.add_argument("--plan", required=True, help="Path to an existing plan directory.")
    run_parser.add_argument("--config", help="Optional config path to run instead of plan defaults.")
    run_parser.add_argument("--smoke", action="store_true", help="Run with a small stopping budget override.")
    run_parser.add_argument("--smoke-evals", type=int, default=200, help="Evaluation budget used when --smoke is set.")
    run_parser.add_argument("--out", help="Output run directory path.")
    run_parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if it already exists.")
    run_parser.add_argument("--json", action="store_true", help="Print JSON output.")

    explain_parser = subcommands.add_parser("explain", help="Summarize an assist plan (and optional run) in user-friendly form.")
    explain_parser.add_argument("--plan", required=True, help="Path to an existing plan directory.")
    explain_parser.add_argument("--run", help="Optional run directory to summarize resolved overrides/report status.")
    explain_parser.add_argument("--json", action="store_true", help="Print JSON output.")

    doctor_parser = subcommands.add_parser("doctor", help="Run environment and provider readiness diagnostics.")
    doctor_parser.add_argument("--json", action="store_true", help="Print JSON output.")

    go_parser = subcommands.add_parser("go", help="Create a plan, apply it, and optionally run smoke in one step.")
    go_parser.add_argument("prompt", help="Prompt text describing intent.")
    go_parser.add_argument(
        "--mode",
        choices=("template", "auto"),
        default="template",
        help="Planning mode (default: template).",
    )
    go_parser.add_argument(
        "--provider",
        choices=("mock", "openai"),
        default="mock",
        help=(
            "Provider name for --mode auto: mock|openai. "
            'OpenAI setup: pip install vamos[openai] (or pip install openai); setx OPENAI_API_KEY "...".'
        ),
    )
    go_parser.add_argument("--template", help="Quickstart template key (optional in interactive terminals).")
    go_parser.add_argument(
        "--problem-type",
        choices=("real", "int", "binary"),
        default="real",
        help="Operator problem type used for catalog generation (default: real).",
    )
    go_parser.add_argument("--out", help="Output plan directory path.")
    go_parser.add_argument("--smoke", action="store_true", help="Run a smoke execution after plan/apply.")
    go_parser.add_argument("--smoke-evals", type=int, default=200, help="Evaluation budget used when --smoke is set.")
    go_parser.add_argument("--overwrite", action="store_true", help="Overwrite project/run directories if they already exist.")
    go_parser.add_argument("--json", action="store_true", help="Print JSON output.")

    return parser


def _render_summary(catalog: dict[str, list[str]], *, problem_type: str) -> str:
    lines = [
        "VAMOS Assist Catalog",
        f"Problem type: {problem_type}",
    ]
    sections: tuple[tuple[str, str], ...] = (
        ("algorithms", "Algorithms"),
        ("kernels", "Kernels"),
        ("crossover_methods", "Crossover methods"),
        ("mutation_methods", "Mutation methods"),
        ("templates", "Templates"),
    )
    for key, label in sections:
        values = catalog[key]
        lines.append(f"{label} ({len(values)}):")
        if values:
            lines.append("  " + ", ".join(values))
        else:
            lines.append("  (none)")
    return "\n".join(lines) + "\n"


def _run_catalog(args: argparse.Namespace) -> None:
    catalog = build_catalog(problem_type=args.problem_type)
    if args.json:
        sys.stdout.write(json.dumps(catalog, indent=2, sort_keys=True))
        sys.stdout.write("\n")
        return
    sys.stdout.write(_render_summary(catalog, problem_type=args.problem_type))


def _run_plan(args: argparse.Namespace) -> None:
    out_dir = Path(args.out) if args.out else None
    provider: PlanProvider | None = None
    if args.mode == "auto":
        if args.provider is None:
            provider = None
        elif args.provider == "mock":
            provider = MockPlanProvider()
        elif args.provider == "openai":
            provider = OpenAIPlanProvider()
        else:
            raise SystemExit(f"Unknown provider '{args.provider}'. Supported providers: mock, openai")
    try:
        plan_dir = create_plan(
            prompt=args.prompt,
            template=args.template,
            problem_type=args.problem_type,
            out_dir=out_dir,
            mode=args.mode,
            provider=provider,
            provider_name=args.provider,
        )
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    metadata_path = plan_dir / "plan.json"
    metadata: dict[str, object] = {}
    if metadata_path.is_file():
        loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            metadata = loaded

    payload = {
        "plan_dir": str(plan_dir),
        "prompt_path": str(plan_dir / "prompt.txt"),
        "catalog_path": str(plan_dir / "catalog.json"),
        "config_path": str(plan_dir / "config.json"),
        "plan_path": str(plan_dir / "plan.json"),
        "mode": args.mode,
    }
    if args.mode == "auto":
        provider_info = metadata.get("provider")
        if isinstance(provider_info, dict):
            provider_name = provider_info.get("name")
            if isinstance(provider_name, str):
                payload["provider"] = provider_name
        if "template" in metadata:
            payload["template"] = metadata["template"]
        if "problem_type" in metadata:
            payload["problem_type"] = metadata["problem_type"]

    if args.json:
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True))
        sys.stdout.write("\n")
        return
    sys.stdout.write(f"Created assist plan at {plan_dir}\n")
    sys.stdout.write(f"Mode: {args.mode}\n")
    if args.mode == "auto":
        if "provider" in payload:
            sys.stdout.write(f"Provider: {payload['provider']}\n")
        if "template" in payload:
            sys.stdout.write(f"Template: {payload['template']}\n")
        if "problem_type" in payload:
            sys.stdout.write(f"Problem type: {payload['problem_type']}\n")
    sys.stdout.write(f"Config: {plan_dir / 'config.json'}\n")
    sys.stdout.write(f"Catalog: {plan_dir / 'catalog.json'}\n")
    sys.stdout.write(f"Metadata: {plan_dir / 'plan.json'}\n")
    sys.stdout.write(f"Next: python -m vamos.experiment.cli.main assist apply --plan {plan_dir}\n")


def _quote_arg(value: str) -> str:
    if any(char.isspace() for char in value):
        escaped = value.replace('"', '\\"')
        return f'"{escaped}"'
    return value


def _run_apply(args: argparse.Namespace) -> None:
    out_dir = Path(args.out) if args.out else None
    project_dir = apply_plan(
        Path(args.plan),
        out_dir=out_dir,
        overwrite=bool(args.overwrite),
    )
    config_path = project_dir / "config.json"
    readme_path = project_dir / "README_run.md"
    recommended = f"vamos --config {_quote_arg(str(config_path))}"
    payload = {
        "project_dir": str(project_dir),
        "config_path": str(config_path),
        "readme_path": str(readme_path),
        "recommended_run_command": recommended,
    }
    if args.json:
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True))
        sys.stdout.write("\n")
        return
    sys.stdout.write(f"Created project directory: {project_dir}\n")
    sys.stdout.write(f"Config: {config_path}\n")
    sys.stdout.write(f"Run: {recommended}\n")


def _run_run(args: argparse.Namespace) -> None:
    summary = run_plan(
        plan_dir=Path(args.plan),
        config_override=Path(args.config) if args.config else None,
        out_dir=Path(args.out) if args.out else None,
        smoke=bool(args.smoke),
        smoke_evals=int(args.smoke_evals),
        overwrite=bool(args.overwrite),
    )
    if args.json:
        sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True))
        sys.stdout.write("\n")
        return
    status = str(summary.get("status", "error"))
    run_dir = summary.get("run_dir")
    resolved_cfg = summary.get("resolved_config_path")
    report_path = summary.get("run_report_path")
    exit_code_value = summary.get("exit_code", 1)
    if isinstance(exit_code_value, bool):
        exit_code = int(exit_code_value)
    elif isinstance(exit_code_value, (int, float, str)):
        try:
            exit_code = int(exit_code_value)
        except ValueError:
            exit_code = 1
    else:
        exit_code = 1
    sys.stdout.write(f"Assist run status: {status} (exit code {exit_code})\n")
    sys.stdout.write(f"Run dir: {run_dir}\n")
    sys.stdout.write(f"Resolved config: {resolved_cfg}\n")
    sys.stdout.write(f"Report: {report_path}\n")
    if status == "ok":
        sys.stdout.write(f"Next: vamos --config {_quote_arg(str(resolved_cfg))}\n")


def _run_explain(args: argparse.Namespace) -> None:
    run_dir = Path(args.run) if args.run else None
    summary = summarize_plan(Path(args.plan), run_dir=run_dir)
    if args.json:
        sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True))
        sys.stdout.write("\n")
        return

    defaults = summary.get("defaults")
    paths = summary.get("paths")
    template = summary.get("template")
    problem_type = summary.get("problem_type")
    commands = summary.get("recommended_next_commands")

    sys.stdout.write("Assist Plan Summary\n")
    if template is not None:
        sys.stdout.write(f"Template: {template}\n")
    if problem_type is not None:
        sys.stdout.write(f"Problem type: {problem_type}\n")

    if isinstance(defaults, dict):
        algorithm = defaults.get("algorithm")
        kernel = defaults.get("kernel")
        budget = defaults.get("stopping_budget")
        output_root = defaults.get("output_root")
        if algorithm is not None:
            sys.stdout.write(f"Algorithm: {algorithm}\n")
        if kernel is not None:
            sys.stdout.write(f"Kernel: {kernel}\n")
        if isinstance(budget, dict):
            budget_key = budget.get("key")
            budget_value = budget.get("value")
            sys.stdout.write(f"Stopping/budget: {budget_key}={budget_value}\n")
        if output_root is not None:
            sys.stdout.write(f"Output root: {output_root}\n")

    if isinstance(paths, dict):
        sys.stdout.write("Artifacts:\n")
        for key in sorted(paths):
            sys.stdout.write(f"  {key}: {paths[key]}\n")

    run_info = summary.get("run")
    if isinstance(run_info, dict):
        status = run_info.get("status", "unknown")
        exit_code = run_info.get("exit_code", "unknown")
        sys.stdout.write(f"Run status: {status} (exit code {exit_code})\n")
        overrides = run_info.get("resolved_overrides")
        if isinstance(overrides, dict):
            defaults_overrides = overrides.get("defaults")
            if isinstance(defaults_overrides, dict) and defaults_overrides:
                sys.stdout.write("Resolved overrides:\n")
                for key in sorted(defaults_overrides):
                    item = defaults_overrides[key]
                    if isinstance(item, dict):
                        sys.stdout.write(f"  defaults.{key}: {item.get('before')} -> {item.get('after')}\n")

    if isinstance(commands, list) and commands:
        sys.stdout.write("Recommended next commands:\n")
        for cmd in commands:
            sys.stdout.write(f"  {cmd}\n")


def _run_doctor(args: argparse.Namespace) -> None:
    report = collect_doctor_report()
    if args.json:
        sys.stdout.write(json.dumps(report, indent=2, sort_keys=True))
        sys.stdout.write("\n")
        return
    sys.stdout.write(format_doctor_report_text(report))


def _run_go(args: argparse.Namespace) -> None:
    try:
        summary = go(
            prompt=args.prompt,
            mode=args.mode,
            provider_name=args.provider,
            template=args.template,
            problem_type=args.problem_type,
            out_dir=Path(args.out) if args.out else None,
            overwrite=bool(args.overwrite),
            smoke=bool(args.smoke),
            smoke_evals=int(args.smoke_evals),
        )
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    if args.json:
        sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True))
        sys.stdout.write("\n")
        return

    plan_dir = summary.get("plan_dir")
    project_dir = summary.get("project_dir")
    sys.stdout.write(f"Created assist plan: {plan_dir}\n")
    sys.stdout.write(f"Created project: {project_dir}\n")

    commands = summary.get("recommended_commands")
    if isinstance(commands, list) and commands:
        sys.stdout.write("Next commands:\n")
        for cmd in commands:
            if isinstance(cmd, str):
                sys.stdout.write(f"  {cmd}\n")

    run_info = summary.get("run")
    if isinstance(run_info, dict):
        status = run_info.get("status", "error")
        run_dir = run_info.get("run_dir")
        sys.stdout.write(f"Smoke run: {status}\n")
        sys.stdout.write(f"Run dir: {run_dir}\n")


def run_assist(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.assist_command == "catalog":
        _run_catalog(args)
        return
    if args.assist_command == "plan":
        _run_plan(args)
        return
    if args.assist_command == "apply":
        _run_apply(args)
        return
    if args.assist_command == "run":
        _run_run(args)
        return
    if args.assist_command == "explain":
        _run_explain(args)
        return
    if args.assist_command == "doctor":
        _run_doctor(args)
        return
    if args.assist_command == "go":
        _run_go(args)
        return
    parser.error(f"Unknown assist subcommand: {args.assist_command}")


__all__ = ["run_assist"]
