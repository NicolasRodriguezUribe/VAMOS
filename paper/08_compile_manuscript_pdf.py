"""
Compile the paper manuscript PDF.

Usage:
  python paper/08_compile_manuscript_pdf.py

Notes:
  - Uses latexmk when available.
  - Runs in paper/manuscript/ and targets main.tex.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    manuscript_dir = Path(__file__).parent / "manuscript"
    tex = manuscript_dir / "main.tex"
    if not tex.is_file():
        raise FileNotFoundError(f"Missing manuscript: {tex}")

    latexmk = shutil.which("latexmk")
    if latexmk:
        cmd = [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex.name]
    else:
        pdflatex = shutil.which("pdflatex")
        if not pdflatex:
            raise RuntimeError("Neither 'latexmk' nor 'pdflatex' was found on PATH.")
        cmd = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex.name]

    print(f"Running: {' '.join(cmd)} (cwd={manuscript_dir})")
    subprocess.run(cmd, cwd=manuscript_dir, check=True)
    print(f"Built: {manuscript_dir / 'main.pdf'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
