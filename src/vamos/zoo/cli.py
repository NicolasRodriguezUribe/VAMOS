from __future__ import annotations

import argparse
from pathlib import Path

from vamos.problem.registry_info import list_problems, get_problem_info
from vamos.problem.registry import make_problem_selection
from vamos.core.runner import run_single
from vamos.core.experiment_config import ExperimentConfig


def _list_cmd(args):
    infos = list_problems()
    for info in sorted(infos, key=lambda x: x.name):
        cats = ",".join(info.categories)
        print(f"{info.name:15} | {cats:12} | n_var={info.default_n_variables} n_obj={info.default_n_objectives} | {info.description}")


def _info_cmd(args):
    info = get_problem_info(args.name)
    if info is None:
        print(f"Problem '{args.name}' not found.")
        return
    print(f"Name: {info.name}")
    print(f"Description: {info.description}")
    print(f"Categories: {', '.join(info.categories)}")
    print(f"Defaults: n_var={info.default_n_variables}, n_obj={info.default_n_objectives}")
    print(f"Tags: {', '.join(info.tags)}")


def _run_cmd(args):
    selection = make_problem_selection(args.problem)
    config = ExperimentConfig(
        output_root=args.output,
        population_size=args.pop_size,
        max_evaluations=args.budget,
        seed=args.seed,
    )
    metrics = run_single(
        args.engine,
        args.algorithm,
        selection,
        config,
        selection_pressure=2,
    )
    print(f"Run finished. HV: {metrics.get('hv')}, output: {metrics.get('output_dir')}")


def build_parser():
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


def main(argv=None):
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
