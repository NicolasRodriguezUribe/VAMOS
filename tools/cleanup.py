#!/usr/bin/env python3
"""
cleanup.py - Maintenance script to clean up temporary files, logs, and empty directories.

Usage:
    python tools/cleanup.py [--dry-run]

Description:
    scans the project root for:
    - Empty directories
    - Temporary files (*.log, *.tmp, *.bak, debug_*.py)
    - Python cache files (__pycache__, .pytest_cache)

    and removes them.
"""

import os
import shutil
import argparse
from pathlib import Path

# Files to verify they are NOT deleted (Safety Check)
CRITICAL_FILES = {
    "pyproject.toml",
    "README.md",
    "src",
    "tests",
    "notebooks",
}

# Patterns to delete
DELETE_PATTERNS = [
    "*.log",
    "*.tmp",
    "*.bak",
    "*.swp",
    "debug_*.py",
    "temp_*.py",
    "*test_output*.txt",
    "arch_test_output*.txt",
]

# Directories to exclude from scanning
EXCLUDE_DIRS = {
    ".git",
    ".venv",
    ".idea",
    ".vscode",
    "node_modules",
}


def clean_empty_dirs(root: Path, dry_run: bool = False):
    """Recursively delete empty directories."""
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        p = Path(dirpath)

        # Skip if in excluded root
        if any(exc in p.parts for exc in EXCLUDE_DIRS):
            continue

        if not dirnames and not filenames:
            if p == root:
                continue
            print(f"[RM DIR] {p}")
            if not dry_run:
                try:
                    p.rmdir()
                except OSError:
                    pass


def clean_files(root: Path, dry_run: bool = False):
    """Delete files matching patterns."""
    for pattern in DELETE_PATTERNS:
        # rglob is recursive
        for path in root.rglob(pattern):
            if path.is_dir():
                continue

            # Skip excluded
            if any(exc in path.parts for exc in EXCLUDE_DIRS):
                continue

            print(f"[RM FILE] {path}")
            if not dry_run:
                try:
                    path.unlink()
                except OSError as e:
                    print(f"Error deleting {path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Project cleanup tool")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be deleted without deleting")
    args = parser.parse_args()

    root = Path(__file__).parent.parent.resolve()
    print(f"Cleaning project root: {root}")
    if args.dry_run:
        print("--- DRY RUN MODE ---")

    clean_files(root, args.dry_run)
    clean_empty_dirs(root, args.dry_run)

    print("Cleanup complete.")


if __name__ == "__main__":
    main()
