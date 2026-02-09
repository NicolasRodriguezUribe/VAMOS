from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vamos.foundation.core.experiment_config import ExperimentConfig


def _configure_cli_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)
    root.setLevel(level)


def _ensure_project_root_on_path() -> None:
    # Allow running via `python src/vamos/cli/main.py` without installing the package.
    module_path = Path(__file__).resolve()
    project_root = module_path.parents[3]
    if __package__ is None or __package__ == "":
        project_root_str = str(project_root)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)


def _dispatch_subcommand(argv: list[str]) -> bool:
    if not argv:
        return False
    command = argv[0]
    if command in {"quickstart", "--quickstart"}:
        from vamos.experiment.cli.quickstart import run_quickstart

        run_quickstart(argv[1:])
        return True
    if command in {"summarize", "summary"}:
        from vamos.experiment.cli.results_cli import run_summarize

        run_summarize(argv[1:])
        return True
    if command in {"ablation"}:
        from vamos.experiment.cli.ablation import run_ablation

        run_ablation(argv[1:])
        return True
    if command in {"assist"}:
        from vamos.assist.cli import run_assist

        run_assist(argv[1:])
        return True
    if command in {"open-results", "open_results"}:
        from vamos.experiment.cli.results_cli import run_open_results

        run_open_results(argv[1:])
        return True
    if command in {"create-problem", "create_problem"}:
        from vamos.experiment.cli.create_problem import run_create_problem

        run_create_problem(argv[1:])
        return True
    return False


def _build_runtime_config(*, args: argparse.Namespace, default_config: ExperimentConfig) -> ExperimentConfig:
    from vamos.foundation.core.experiment_config import ExperimentConfig

    return ExperimentConfig(
        title=default_config.title,
        output_root=args.output_root,
        population_size=args.population_size,
        offspring_population_size=args.offspring_population_size,
        max_evaluations=args.max_evaluations,
        seed=args.seed,
        eval_strategy=args.eval_strategy,
        n_workers=args.n_workers,
        live_viz=args.live_viz,
        live_viz_interval=args.live_viz_interval,
        live_viz_max_points=args.live_viz_max_points,
    )


def _run_standard_cli(argv: Sequence[str] | None = None) -> int:
    from vamos.experiment.cli import parse_args
    from vamos.experiment.cli.preflight import run_preflight_checks
    from vamos.experiment.presentation import run_experiments_from_args
    from vamos.foundation.core.experiment_config import ExperimentConfig

    default_config = ExperimentConfig()
    args = parse_args(default_config, argv=argv)
    if getattr(args, "quiet", False):
        _configure_cli_logging(logging.WARNING)
    elif getattr(args, "verbose", False):
        _configure_cli_logging(logging.DEBUG)

    config = _build_runtime_config(args=args, default_config=default_config)
    if not getattr(args, "no_preflight", False):
        run_preflight_checks(args)
    run_experiments_from_args(args, config)
    return 0


def _config_only_path(argv: Sequence[str]) -> str | None:
    from vamos.experiment.cli.args import build_pre_parser

    pre_parser = build_pre_parser()
    pre_args, remaining = pre_parser.parse_known_args(argv)
    config_path = getattr(pre_args, "config", None)
    if not isinstance(config_path, str) or not config_path:
        return None
    if getattr(pre_args, "validate_config", False):
        return None
    if remaining:
        return None
    return config_path


def run_from_config_path(config_path: str) -> int:
    _ensure_project_root_on_path()
    _configure_cli_logging()
    try:
        return _run_standard_cli(argv=["--config", config_path])
    except SystemExit as exc:
        if isinstance(exc.code, int):
            return exc.code
        return 1


def main() -> None:
    _ensure_project_root_on_path()
    _configure_cli_logging()
    argv = sys.argv[1:]
    if _dispatch_subcommand(list(argv)):
        return

    config_only = _config_only_path(argv)
    if config_only is not None:
        exit_code = run_from_config_path(config_only)
        if exit_code != 0:
            raise SystemExit(exit_code)
        return

    _run_standard_cli(argv=argv)


if __name__ == "__main__":
    main()
