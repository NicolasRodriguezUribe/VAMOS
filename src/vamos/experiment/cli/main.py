from __future__ import annotations

import logging
import sys
from pathlib import Path


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
    if command in {"open-results", "open_results"}:
        from vamos.experiment.cli.results_cli import run_open_results

        run_open_results(argv[1:])
        return True
    return False


def main() -> None:
    _ensure_project_root_on_path()
    _configure_cli_logging()
    if _dispatch_subcommand(sys.argv[1:]):
        return
    from vamos.experiment.cli import parse_args
    from vamos.experiment.cli.preflight import run_preflight_checks
    from vamos.experiment.presentation import run_experiments_from_args
    from vamos.foundation.core.experiment_config import ExperimentConfig

    default_config = ExperimentConfig()
    args = parse_args(default_config)
    if getattr(args, "quiet", False):
        _configure_cli_logging(logging.WARNING)
    elif getattr(args, "verbose", False):
        _configure_cli_logging(logging.DEBUG)
    config = ExperimentConfig(
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
    if not getattr(args, "no_preflight", False):
        run_preflight_checks(args)
    run_experiments_from_args(args, config)


if __name__ == "__main__":
    main()
