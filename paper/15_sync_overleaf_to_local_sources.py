"""
Sync Overleaf project sources into local manuscript sources (excluding PDF/build artifacts).

This script fetches an Overleaf Git remote branch, checks it out in a temporary
worktree, and mirrors its source files into a local destination directory.

Usage:
  python paper/15_sync_overleaf_to_local_sources.py
  python paper/15_sync_overleaf_to_local_sources.py --dry-run
  python paper/15_sync_overleaf_to_local_sources.py --remote overleaf --branch master
"""

from __future__ import annotations

import argparse
import filecmp
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


EXCLUDE_FILENAMES = {
    ".git",
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
        description="Sync Overleaf project sources into local manuscript sources."
    )
    parser.add_argument(
        "--remote",
        default="overleaf",
        help="Git remote name for the Overleaf project (default: overleaf).",
    )
    parser.add_argument(
        "--branch",
        default="master",
        help="Branch on the Overleaf remote to pull from (default: master).",
    )
    parser.add_argument(
        "--dst",
        default=str(Path("paper") / "manuscript"),
        help="Destination directory to sync into (default: paper/manuscript).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned file changes without writing to disk.",
    )
    return parser.parse_args()


def _run(cmd: list[str], *, cwd: Path) -> None:
    printable = shlex.join(cmd)
    print(f"+ {printable}  (cwd={cwd})")
    subprocess.run(cmd, cwd=cwd, text=True, check=True)


def _run_capture(cmd: list[str], *, cwd: Path) -> str:
    printable = shlex.join(cmd)
    print(f"+ {printable}  (cwd={cwd})")
    proc = subprocess.run(cmd, cwd=cwd, text=True, check=True, capture_output=True)
    return proc.stdout.strip()


def _should_exclude_rel(rel: Path) -> bool:
    if any(part in EXCLUDE_DIRS for part in rel.parts):
        return True

    name = rel.name
    if name in EXCLUDE_FILENAMES:
        return True

    if any(name.endswith(suf) for suf in EXCLUDE_NAME_SUFFIXES):
        return True

    if Path(name).suffix in EXCLUDE_SUFFIXES:
        return True

    return False


def _collect_paths(root: Path) -> tuple[set[Path], set[Path]]:
    dirs: set[Path] = set()
    files: set[Path] = set()

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]

        current = Path(dirpath)
        rel_dir = current.relative_to(root)
        if rel_dir != Path("."):
            dirs.add(rel_dir)

        for filename in filenames:
            rel_file = (current / filename).relative_to(root)
            if _should_exclude_rel(rel_file):
                continue
            files.add(rel_file)

    return dirs, files


def _build_plan(
    src_dir: Path,
    dst_dir: Path,
) -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    src_dirs, src_files = _collect_paths(src_dir)
    dst_dirs, dst_files = _collect_paths(dst_dir)

    files_to_delete = sorted(dst_files - src_files)

    files_to_add: list[Path] = []
    files_to_update: list[Path] = []
    for rel_file in sorted(src_files):
        src_file = src_dir / rel_file
        dst_file = dst_dir / rel_file
        if not dst_file.exists():
            files_to_add.append(rel_file)
            continue
        if not filecmp.cmp(src_file, dst_file, shallow=False):
            files_to_update.append(rel_file)

    dirs_to_prune = sorted(
        dst_dirs - src_dirs,
        key=lambda path: len(path.parts),
        reverse=True,
    )

    return files_to_add, files_to_update, files_to_delete, dirs_to_prune


def _print_preview(label: str, files: list[Path], *, limit: int = 10) -> None:
    if not files:
        return
    print(f"{label} ({len(files)}):")
    for rel in files[:limit]:
        print(f"  - {rel.as_posix()}")
    if len(files) > limit:
        print(f"  ... {len(files) - limit} more")


def _apply_plan(
    *,
    src_dir: Path,
    dst_dir: Path,
    files_to_add: list[Path],
    files_to_update: list[Path],
    files_to_delete: list[Path],
    dirs_to_prune: list[Path],
) -> None:
    for rel_file in files_to_delete:
        dst_file = dst_dir / rel_file
        if dst_file.exists():
            dst_file.unlink()

    for rel_file in [*files_to_add, *files_to_update]:
        src_file = src_dir / rel_file
        dst_file = dst_dir / rel_file
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)

    for rel_dir in dirs_to_prune:
        dst_subdir = dst_dir / rel_dir
        if not dst_subdir.is_dir():
            continue
        try:
            dst_subdir.rmdir()
        except OSError:
            continue


def main() -> None:
    args = _parse_args()

    repo_root = Path(_run_capture(["git", "rev-parse", "--show-toplevel"], cwd=Path.cwd()))
    dst_dir = (repo_root / args.dst).resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)

    available_remotes = {
        line.strip()
        for line in _run_capture(["git", "remote"], cwd=repo_root).splitlines()
        if line.strip()
    }
    if args.remote not in available_remotes:
        remotes_text = ", ".join(sorted(available_remotes)) if available_remotes else "(none)"
        raise RuntimeError(
            f"Git remote '{args.remote}' was not found. "
            f"Available remotes: {remotes_text}. "
            "Add the Overleaf remote first (for example: git remote add overleaf <url>)."
        )

    _run(["git", "fetch", args.remote, args.branch], cwd=repo_root)

    temp_dir = Path(tempfile.mkdtemp(prefix="vamos_overleaf_pull_"))
    temp_branch = f"overleaf-pull-{os.getpid()}-{int(time.time())}"
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

        src_dir = temp_dir
        main_tex = src_dir / "main.tex"
        if not main_tex.is_file():
            raise FileNotFoundError(
                f"Missing main.tex in Overleaf branch root: {main_tex}"
            )

        files_to_add, files_to_update, files_to_delete, dirs_to_prune = _build_plan(
            src_dir=src_dir,
            dst_dir=dst_dir,
        )

        print(
            "Plan: "
            f"{len(files_to_add)} add, "
            f"{len(files_to_update)} update, "
            f"{len(files_to_delete)} delete, "
            f"{len(dirs_to_prune)} empty dirs to prune."
        )
        _print_preview("Add", files_to_add)
        _print_preview("Update", files_to_update)
        _print_preview("Delete", files_to_delete)

        if args.dry_run:
            print("Dry run complete. No local files were modified.")
            return

        _apply_plan(
            src_dir=src_dir,
            dst_dir=dst_dir,
            files_to_add=files_to_add,
            files_to_update=files_to_update,
            files_to_delete=files_to_delete,
            dirs_to_prune=dirs_to_prune,
        )
        print("Pulled Overleaf sources into local manuscript directory successfully.")
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
