import json
from pathlib import Path

from vamos.experiment.cli import results_cli


def _write_run(base: Path, *, problem: str, algorithm: str, engine: str, seed: int, timestamp: str) -> Path:
    run_dir = base / problem.upper() / algorithm / engine / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "timestamp": timestamp,
        "algorithm": algorithm,
        "backend": engine,
        "seed": seed,
        "problem": {"key": problem, "label": problem.upper()},
        "metrics": {
            "time_ms": 1.23,
            "evaluations": 10,
            "evals_per_sec": 8.1,
            "termination": "max_evaluations",
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return run_dir


def test_cli_summarize_latest(capsys, tmp_path):
    _write_run(
        tmp_path,
        problem="zdt1",
        algorithm="nsgaii",
        engine="numpy",
        seed=1,
        timestamp="2026-01-01T00:00:00+00:00",
    )
    results_cli.run_summarize(["--results", str(tmp_path), "--latest"])
    out = capsys.readouterr().out
    assert "zdt1" in out


def test_cli_summarize_json_format(capsys, tmp_path):
    _write_run(
        tmp_path,
        problem="zdt1",
        algorithm="nsgaii",
        engine="numpy",
        seed=1,
        timestamp="2026-01-01T00:00:00+00:00",
    )
    results_cli.run_summarize(["--results", str(tmp_path), "--format", "json", "--latest"])
    out = capsys.readouterr().out
    data = json.loads(out)
    assert isinstance(data, list)
    assert data[0]["problem"] == "zdt1"


def test_cli_open_results_picks_latest(capsys, tmp_path):
    older = _write_run(
        tmp_path,
        problem="zdt1",
        algorithm="nsgaii",
        engine="numpy",
        seed=1,
        timestamp="2026-01-01T00:00:00+00:00",
    )
    newer = _write_run(
        tmp_path,
        problem="zdt1",
        algorithm="nsgaii",
        engine="numpy",
        seed=2,
        timestamp="2026-01-02T00:00:00+00:00",
    )
    results_cli.run_open_results(["--results", str(tmp_path)])
    out = capsys.readouterr().out
    assert str(newer) in out
    assert str(older) not in out
