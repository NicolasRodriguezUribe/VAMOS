from __future__ import annotations

import argparse
import logging

from vamos.foundation.problem.registry_info import list_problems, get_problem_info
from vamos.experiment.runner import run_experiment
from vamos.foundation.core.experiment_config import ExperimentConfig


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def _configure_cli_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
    root.setLevel(level)


def _list_cmd(args: argparse.Namespace) -> None:
    infos = list_problems()
    for info in sorted(infos, key=lambda x: x.name):
        cats = ",".join(info.categories)
        _logger().info(
            "%-15s | %-12s | n_var=%s n_obj=%s | %s",
            info.name,
            cats,
            info.default_n_variables,
            info.default_n_objectives,
            info.description,
        )


def _info_cmd(args: argparse.Namespace) -> None:
    info = get_problem_info(args.name)
    if info is None:
        _logger().warning("Problem '%s' not found.", args.name)
        return
    _logger().info("Name: %s", info.name)
    _logger().info("Description: %s", info.description)
    _logger().info("Categories: %s", ", ".join(info.categories))
    _logger().info(
        "Defaults: n_var=%s, n_obj=%s",
        info.default_n_variables,
        info.default_n_objectives,
    )
    _logger().info("Tags: %s", ", ".join(info.tags))


def _run_cmd(args: argparse.Namespace) -> None:
    config = ExperimentConfig(
        output_root=args.output,
        population_size=args.pop_size,
        max_evaluations=args.budget,
        seed=args.seed,
    )
    metrics = run_experiment(
        algorithm=args.algorithm,
        problem=args.problem,
        engine=args.engine,
        config=config,
        selection_pressure=2,
    )
    _logger().info("Run finished. HV: %s, output: %s", metrics.get("hv"), metrics.get("output_dir"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VAMOS problem zoo browser.")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("list", help="List available problems")

    info_p = sub.add_parser("info", help="Show detailed info for a problem")
    info_p.add_argument("name")

    run_p = sub.add_parser("run", help="Quick run a problem with defaults")
    run_p.add_argument("problem")
    run_p.add_argument("--algorithm", default="nsgaii")
    run_p.add_argument("--engine", default="numpy")
    run_p.add_argument("--budget", type=int, default=5000)
    run_p.add_argument("--pop-size", type=int, default=50)
    run_p.add_argument("--seed", type=int, default=0)
    run_p.add_argument("--output", default="zoo_runs")
    return p


def main(argv: list[str] | None = None) -> None:
    _configure_cli_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd is None:
        args.cmd = "list"
    if args.cmd == "list":
        _list_cmd(args)
    elif args.cmd == "info":
        _info_cmd(args)
    elif args.cmd == "run":
        _run_cmd(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
