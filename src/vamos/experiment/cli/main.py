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


def main() -> None:
    _ensure_project_root_on_path()
    from vamos.experiment.cli import parse_args
    from vamos.experiment.runner import run_experiments_from_args
    from vamos.foundation.core.experiment_config import ExperimentConfig

    _configure_cli_logging()
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
    run_experiments_from_args(args, config)


if __name__ == "__main__":
    main()
