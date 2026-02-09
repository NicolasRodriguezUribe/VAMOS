"""
CLI for ``vamos profile``.
"""

from __future__ import annotations

import argparse
import logging
import sys


def main() -> int:
    """Entry point for ``vamos profile`` CLI."""
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(prog="vamos profile", description="Profile VAMOS optimization across different backends")
    parser.add_argument("--problem", "-p", default="zdt1", help="Problem to benchmark (default: zdt1)")
    parser.add_argument("--engines", "-e", default="numpy", help="Comma-separated list of engines (default: numpy)")
    parser.add_argument("--budget", "-b", type=int, default=2000, help="Evaluation budget (default: 2000)")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output", "-o", help="Output CSV file path (optional)")
    parser.add_argument("--no-hv", action="store_true", help="Skip hypervolume computation")

    args = parser.parse_args()

    # Parse engines
    engines = [e.strip() for e in args.engines.split(",")]

    # Run profiler
    from vamos.experiment.profiler.runner import run_profile

    report = run_profile(problem=args.problem, engines=engines, budget=args.budget, seed=args.seed, compute_hv=not args.no_hv)

    # Output
    report.print_summary()

    if args.output:
        report.to_csv(args.output)
        logger.info("Results saved to: %s", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
