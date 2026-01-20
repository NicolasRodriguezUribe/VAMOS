"""
Sync the Overleaf project with local manuscript sources (excluding the PDF).

This script updates an Overleaf Git remote without force-pushing:
it fetches the remote branch, creates a temporary worktree at that remote HEAD,
replaces the project contents with the local `paper/manuscript/` sources (minus
build artifacts and `main.pdf`), commits, and pushes a fast-forward update.

Usage:
  python paper/11_sync_overleaf_sources.py
  python paper/11_sync_overleaf_sources.py --remote overleaf --branch master
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


EXCLUDE_FILENAMES = {
    "main.pdf",
    ".DS_Store",
    "Thumbs.db",
}

EXCLUDE_SUFFIXES = {
    ".aux",
    ".acn",
    ".acr",
    ".alg",
    ".bbl",
    ".bcf",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".glo",
    ".glg",
    ".gls",
    ".ist",
    ".lof",
    ".log",
    ".lot",
    ".nav",
    ".out",
    ".run.xml",
    ".snm",
    ".spl",
    ".synctex",
    ".toc",
    ".vrb",
    ".xdv",
}

EXCLUDE_NAME_SUFFIXES = {
    ".synctex.gz",
}

EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync Overleaf project with local manuscript sources (excluding the PDF)."
    )
    parser.add_argument(
        "--remote",
        default="overleaf",
        help="Git remote name for the Overleaf project (default: overleaf).",
    )
    parser.add_argument(
        "--branch",
        default="master",
        help="Branch on the Overleaf remote to update (default: master).",
    )
    parser.add_argument(
        "--src",
        default=str(Path("paper") / "manuscript"),
        help="Source directory to sync (default: paper/manuscript).",
    )
    parser.add_argument(
        "--message",
        default="Sync manuscript sources",
        help="Commit message for the Overleaf update.",
    )
    return parser.parse_args()


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    printable = shlex.join(cmd)
    print(f"+ {printable}  (cwd={cwd})")
    return subprocess.run(cmd, cwd=cwd, text=True, check=True)


def _run_capture(cmd: list[str], *, cwd: Path) -> str:
    printable = shlex.join(cmd)
    print(f"+ {printable}  (cwd={cwd})")
    proc = subprocess.run(cmd, cwd=cwd, text=True, check=True, capture_output=True)
    return proc.stdout.strip()


def _has_staged_changes(*, cwd: Path) -> bool:
    proc = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=cwd,
        text=True,
    )
    return proc.returncode != 0


def _should_exclude(path: Path) -> bool:
    if path.is_dir():
        return path.name in EXCLUDE_DIRS

    if path.name in EXCLUDE_FILENAMES:
        return True

    if any(path.name.endswith(suf) for suf in EXCLUDE_NAME_SUFFIXES):
        return True

    if path.suffix in EXCLUDE_SUFFIXES:
        return True

    return False


def _copy_sources(src_dir: Path, dst_dir: Path) -> None:
    for src_path in src_dir.rglob("*"):
        if _should_exclude(src_path):
            continue

        rel = src_path.relative_to(src_dir)
        dst_path = dst_dir / rel

        if src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
            continue

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)


def main() -> None:
    args = _parse_args()

    repo_root = Path(_run_capture(["git", "rev-parse", "--show-toplevel"], cwd=Path.cwd()))
    src_dir = (repo_root / args.src).resolve()
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Missing source dir: {src_dir}")

    main_tex = src_dir / "main.tex"
    if not main_tex.is_file():
        raise FileNotFoundError(f"Missing main.tex in source dir: {main_tex}")

    _run(["git", "fetch", args.remote, args.branch], cwd=repo_root)

    temp_dir = Path(tempfile.mkdtemp(prefix="vamos_overleaf_sync_"))
    temp_branch = f"overleaf-sync-{os.getpid()}-{int(time.time())}"
    try:
        _run(
            [
                "git",
                "worktree",
                "add",
                "-B",
                temp_branch,
                str(temp_dir),
                f"{args.remote}/{args.branch}",
            ],
            cwd=repo_root,
        )

        _run(["git", "rm", "-r", "--quiet", "--ignore-unmatch", "."], cwd=temp_dir)
        _copy_sources(src_dir, temp_dir)

        _run(["git", "add", "-A"], cwd=temp_dir)

        if not _has_staged_changes(cwd=temp_dir):
            print("No changes to push (Overleaf already up to date).")
            return

        _run(["git", "commit", "-m", args.message], cwd=temp_dir)
        _run(["git", "push", args.remote, f"HEAD:{args.branch}"], cwd=temp_dir)

        print("Pushed sources to Overleaf successfully.")
    finally:
        try:
            _run(["git", "worktree", "remove", str(temp_dir)], cwd=repo_root)
        except Exception:
            pass

        try:
            _run(["git", "branch", "-D", temp_branch], cwd=repo_root)
        except Exception:
            pass

        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

