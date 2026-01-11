"""
Publication-Ready Outputs Demo.

Demonstrates how to generate LaTeX tables and interactive plots
from optimization results.
"""

from __future__ import annotations
from vamos.api import optimize


def main():
    print("Running optimization...")
    result = optimize("zdt1", algorithm="nsgaii", budget=3000, seed=42)

    # 1. Generate LaTeX table
    print("\n=== LaTeX Table ===")
    latex_code = result.to_latex(caption="NSGA-II Performance on ZDT1", label="tab:zdt1_nsgaii")
    print(latex_code)

    # Save to file
    with open("zdt1_table.tex", "w") as f:
        f.write(latex_code)
    print("\nSaved table to 'zdt1_table.tex'")

    # 2. Interactive Exploration (uncomment to launch browser)
    # print("\nLaunching interactive dashboard...")
    # result.explore(title="ZDT1 Interactive Analysis")


if __name__ == "__main__":
    main()
