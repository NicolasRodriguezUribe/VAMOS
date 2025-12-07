from __future__ import annotations

import sys
from pathlib import Path

# Allow running via `python src/vamos/main.py` without installing the package.
module_path = Path(__file__).resolve()
project_root = module_path.parents[2]
if __package__ is None or __package__ == "":
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from vamos.cli import parse_args
from vamos.runner import ExperimentConfig, run_from_args


def main():
    default_config = ExperimentConfig()
    args = parse_args(default_config)
    config = ExperimentConfig(
        title=default_config.title,
        output_root=args.output_root,
        population_size=args.population_size,
        offspring_population_size=args.offspring_population_size,
        max_evaluations=args.max_evaluations,
        seed=args.seed,
        eval_backend=args.eval_backend,
        n_workers=args.n_workers,
        live_viz=args.live_viz,
        live_viz_interval=args.live_viz_interval,
        live_viz_max_points=args.live_viz_max_points,
    )
    run_from_args(args, config)


if __name__ == "__main__":
    main()
