from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]


def find_seed_dir(root: Path) -> Path:
    for path in root.rglob("seed_*"):
        if path.is_dir() and (path / "metadata.json").exists():
            return path
    raise FileNotFoundError(f"No seed dir with metadata.json found under: {root}")


def test_hv_archive_artifacts_and_metadata():
    cfg = REPO / "experiments" / "configs" / "hv_archive_validation_slice.yml"
    assert cfg.exists()

    cmd = [sys.executable, "-m", "vamos.experiment.cli.main", "--config", str(cfg)]
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout

    out_root = REPO / "results" / "hv_archive_slice"
    assert out_root.exists()

    seed_dir = find_seed_dir(out_root)
    hv = seed_dir / "hv_trace.csv"
    ar = seed_dir / "archive_stats.csv"
    md = seed_dir / "metadata.json"

    assert hv.exists(), f"Missing hv_trace.csv in {seed_dir}"
    assert ar.exists(), f"Missing archive_stats.csv in {seed_dir}"
    assert md.exists()

    meta = json.loads(md.read_text(encoding="utf-8"))
    assert "stopping" in meta, "metadata missing stopping"
    assert "archive" in meta, "metadata missing archive"
    assert meta["stopping"].get("enabled") in (True, False)
    assert meta["archive"].get("enabled") in (True, False)

    hv_lines = hv.read_text(encoding="utf-8").splitlines()
    assert len(hv_lines) >= 3
    assert hv_lines[0].startswith("evals,hv,hv_delta,stop_flag,reason")

    ar_lines = ar.read_text(encoding="utf-8").splitlines()
    assert len(ar_lines) >= 2
    assert ar_lines[0].startswith("evals,archive_size,inserted,pruned,prune_reason")
