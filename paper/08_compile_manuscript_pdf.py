"""
Compile the paper manuscript PDF.

Usage:
  python paper/08_compile_manuscript_pdf.py
  python paper/08_compile_manuscript_pdf.py --no-sync  (or --no-sync-overleaf)

Notes:
  - Uses latexmk when available.
  - Runs in paper/manuscript/ and targets main.tex.
  - Writes the compiled PDF and auxiliary files below ignored paper/build/ by default.
  - By default, it also syncs sources to Overleaf (not main.pdf).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile the paper manuscript PDF.")
    parser.add_argument(
        "--sync-overleaf",
        dest="sync_overleaf",
        action="store_true",
        default=True,
        help="After a successful build, sync manuscript sources to the Overleaf Git remote (default).",
    )
    parser.add_argument(
        "--no-sync-overleaf",
        "--no-sync",
        dest="sync_overleaf",
        action="store_false",
        help="Do not sync manuscript sources to Overleaf after building.",
    )
    parser.add_argument(
        "--overleaf-remote",
        default="overleaf",
        help="Git remote name for the Overleaf project (default: overleaf).",
    )
    parser.add_argument(
        "--overleaf-branch",
        default="master",
        help="Branch on the Overleaf remote to update (default: master).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "build",
        help="Ignored directory for the compiled PDF and auxiliary files (default: paper/build).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing compiled PDF in the output directory.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    manuscript_dir = Path(__file__).parent / "manuscript"
    tex = manuscript_dir / "main.tex"
    if not tex.is_file():
        raise FileNotFoundError(f"Missing manuscript: {tex}")
    output_dir = args.output_dir.expanduser().resolve()
    output_pdf = output_dir / "main.pdf"
    if output_pdf.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing paper build: {output_pdf}. Pass --overwrite to replace it.")
    output_dir.mkdir(parents=True, exist_ok=True)

    latexmk = shutil.which("latexmk")
    if latexmk:
        cmd = [latexmk, "-pdf", f"-outdir={output_dir}", "-interaction=nonstopmode", "-halt-on-error", tex.name]
    else:
        pdflatex = shutil.which("pdflatex")
        if not pdflatex:
            raise RuntimeError("Neither 'latexmk' nor 'pdflatex' was found on PATH.")
        cmd = [pdflatex, f"-output-directory={output_dir}", "-interaction=nonstopmode", "-halt-on-error", tex.name]

    print(f"Running: {' '.join(cmd)} (cwd={manuscript_dir})")
    subprocess.run(cmd, cwd=manuscript_dir, check=True)
    print(f"Built: {output_pdf}")

    if args.sync_overleaf:
        repo_root = Path(__file__).resolve().parent.parent
        sync_script = repo_root / "paper" / "11_sync_overleaf_sources.py"
        if not sync_script.is_file():
            raise FileNotFoundError(f"Missing sync script: {sync_script}")

        subprocess.run(
            [
                sys.executable,
                str(sync_script),
                "--remote",
                args.overleaf_remote,
                "--branch",
                args.overleaf_branch,
            ],
            check=True,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
