"""
Shared result layout helpers for experiment outputs.

Standard layout (relative to `output_root`, defaults to `results/`):
    <problem_label>/<algorithm>/<engine>/seed_<seed>/
        FUN.csv
        X.csv (optional)
        G.csv (optional)
        ARCHIVE_FUN.csv / ARCHIVE_X.csv / ARCHIVE_G.csv (optional)
        metadata.json
        resolved_config.json
        time.txt
"""

from __future__ import annotations

from pathlib import Path

RESULT_FILES = {
    "fun": "FUN.csv",
    "x": "X.csv",
    "g": "G.csv",
    "archive_fun": "ARCHIVE_FUN.csv",
    "archive_x": "ARCHIVE_X.csv",
    "archive_g": "ARCHIVE_G.csv",
    "metadata": "metadata.json",
    "resolved_config": "resolved_config.json",
    "time": "time.txt",
}


def standard_run_dir(
    *,
    problem_label: str,
    algorithm: str,
    engine: str,
    seed: int,
    output_root: str | Path = "results",
) -> Path:
    base = Path(output_root)
    return base / str(problem_label).upper() / algorithm.lower() / engine.lower() / f"seed_{seed}"
