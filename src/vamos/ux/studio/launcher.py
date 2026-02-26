"""Launcher for VAMOS Studio via the Streamlit runtime."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _studio_app_path() -> Path:
    return Path(__file__).resolve().with_name("app.py")


def _build_streamlit_command(study_dir: str, streamlit_args: Sequence[str]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(_studio_app_path()),
        *streamlit_args,
        "--",
        "--study-dir",
        study_dir,
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vamos studio", description="Launch VAMOS Studio (Streamlit).")
    parser.add_argument("--study-dir", default="results", help="Path to a StudyRunner output directory.")
    args, streamlit_args = parser.parse_known_args(argv)

    try:
        import streamlit  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("VAMOS Studio requires the 'studio' extra: pip install -e \".[studio]\"") from exc

    command = _build_streamlit_command(str(args.study_dir), streamlit_args)
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


__all__ = ["main"]
