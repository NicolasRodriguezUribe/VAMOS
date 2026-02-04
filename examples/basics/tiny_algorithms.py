"""
Tiny algorithm examples: NSGA-II, MOEA/D, SMS-EMOA.

Usage:
    python examples/basics/tiny_algorithms.py
"""

from __future__ import annotations

from vamos import optimize


def run_nsgaii() -> None:
    print("\n=== NSGA-II (tiny) ===")
    result = optimize("zdt1", algorithm="nsgaii", pop_size=40)
    print(f"NSGA-II solutions: {len(result)}")


def run_moead() -> None:
    print("\n=== MOEA/D (tiny) ===")
    result = optimize("zdt1", algorithm="moead", pop_size=40)
    print(f"MOEA/D solutions: {len(result)}")


def run_smsemoa() -> None:
    print("\n=== SMS-EMOA (tiny) ===")
    result = optimize("zdt1", algorithm="smsemoa", pop_size=40)
    print(f"SMS-EMOA solutions: {len(result)}")


def main() -> None:
    run_nsgaii()
    run_moead()
    run_smsemoa()


if __name__ == "__main__":
    main()
