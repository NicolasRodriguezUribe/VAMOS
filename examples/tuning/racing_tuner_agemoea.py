"""
AGE-MOEA racing tuner (wrapper).

This thin wrapper forwards to the generic racing tuner with AGE-MOEA defaults.

Usage:
    python examples/tuning/racing_tuner_agemoea.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import runpy


def _inject_default(args: list[str], flag: str, value: str | None = None) -> list[str]:
    if flag in args:
        return args
    if value is None:
        return args + [flag]
    return args + [flag, value]


def main() -> None:
    argv = sys.argv[1:]
    argv = _inject_default(argv, "--algorithm", "agemoea")
    argv = _inject_default(argv, "--multi-fidelity")
    argv = _inject_default(argv, "--fidelity-levels", "500,1000,1500")
    argv = _inject_default(argv, "--tune-budget", "12")
    argv = _inject_default(argv, "--n-seeds", "2")
    argv = _inject_default(argv, "--seed", "0")
    sys.argv = [sys.argv[0], *argv]
    generic_path = Path(__file__).resolve().with_name("racing_tuner_generic.py")
    runpy.run_path(str(generic_path), run_name="__main__")


if __name__ == "__main__":
    main()
