"""Launcher for VAMOS Studio via the Panel runtime."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path


def _studio_app_path() -> Path:
    return Path(__file__).resolve().with_name("app.py")


def _build_panel_command(study_dir: str, panel_args: Sequence[str]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "panel",
        "serve",
        str(_studio_app_path()),
        "--show",
        "--autoreload",
        f"--args=--study-dir={study_dir}",
        *panel_args,
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vamos studio", description="Launch VAMOS Studio (Panel).")
    parser.add_argument("--study-dir", default="results", help="Path to a canonical StudyManifest directory.")
    parser.add_argument("--port", type=int, default=5006, help="Port for the Panel server.")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open a browser tab.")
    args, panel_args = parser.parse_known_args(argv)

    try:
        import panel  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError("VAMOS Studio requires the 'studio' extra: pip install -e \".[studio]\"") from exc

    cmd = [
        sys.executable,
        "-m",
        "panel",
        "serve",
        str(_studio_app_path()),
        f"--port={args.port}",
        "--autoreload",
        f"--args=--study-dir={args.study_dir}",
        *panel_args,
    ]
    if not args.no_browser:
        cmd.append("--show")

    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


__all__ = ["main"]
