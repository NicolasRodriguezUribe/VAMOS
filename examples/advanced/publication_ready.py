"""
Publication-Ready Outputs Demo.

Demonstrates how to generate LaTeX tables and interactive plots
from optimization results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vamos import optimize
from vamos.ux.api import result_to_latex


DEFAULT_OUTPUT = Path("artifacts/publication/zdt1_table.tex")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination for the generated LaTeX table.")
    parser.add_argument("--max-evaluations", type=int, default=3000, help="Optimization evaluation budget.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {output}. Pass --overwrite to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)
    print("Running optimization...")
    result = optimize("zdt1", algorithm="nsgaii", max_evaluations=args.max_evaluations, seed=42)

    # 1. Generate LaTeX table
    print("\n=== LaTeX Table ===")
    latex_code = result_to_latex(result, caption="NSGA-II Performance on ZDT1", label="tab:zdt1_nsgaii")
    print(latex_code)

    output.write_text(latex_code, encoding="utf-8")
    print(f"\nSaved table to {output}")

    # 2. Interactive Exploration (uncomment to launch browser)
    # print("\nLaunching interactive dashboard...")
    # from vamos.ux.api import explore_result_front
    # explore_result_front(result, title="ZDT1 Interactive Analysis")


if __name__ == "__main__":
    main()
