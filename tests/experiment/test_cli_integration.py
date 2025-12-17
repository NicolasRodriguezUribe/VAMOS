import sys

import numpy as np

from vamos.foundation.core import runner


def test_cli_runs_and_writes_artifacts(monkeypatch, tmp_path):
    """
    Minimal end-to-end check: CLI -> runner -> algorithm -> artifacts.
    Uses a tiny NSGA-II run on ZDT1 to keep runtime small.
    """
    output_root = tmp_path / "results"
    monkeypatch.setenv("VAMOS_OUTPUT_ROOT", str(output_root))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    # Skip plotting in this fast-path integration check.
    monkeypatch.setattr(runner.plotting, "plot_pareto_front", lambda *args, **kwargs: None)

    argv = [
        "prog",
        "--problem",
        "zdt1",
        "--algorithm",
        "nsgaii",
        "--engine",
        "numpy",
        "--population-size",
        "6",
        "--offspring-population-size",
        "6",
        "--max-evaluations",
        "10",
        "--seed",
        "1",
    ]
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setattr(sys, "argv", argv)

    from vamos.experiment.cli.main import main

    main()

    run_dir = output_root / "ZDT1" / "nsgaii" / "numpy" / "seed_1"
    fun_path = run_dir / "FUN.csv"
    metadata_path = run_dir / "metadata.json"
    assert fun_path.exists(), "FUN.csv was not produced by the CLI run"
    assert metadata_path.exists(), "metadata.json was not produced by the CLI run"
    fun = np.loadtxt(fun_path, delimiter=",")
    assert fun.shape[0] > 0
