import sys

import vamos
from vamos.ux.visualization import plotting


def test_cli_runs_and_writes_artifacts(monkeypatch, tmp_path):
    """
    Minimal end-to-end check: CLI -> runner -> algorithm -> artifacts.
    Uses a tiny NSGA-II run on ZDT1 to keep runtime small.
    """
    output_root = tmp_path / "results"
    monkeypatch.setenv("VAMOS_OUTPUT_ROOT", str(output_root))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    # Skip plotting in this fast-path integration check.
    monkeypatch.setattr(plotting, "plot_pareto_front", lambda *args, **kwargs: None)

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
    assert {path.name for path in run_dir.iterdir()} == {"manifest.json", "result.npz", "environment.json"}
    stored = vamos.load_run(run_dir, verify="all")
    assert stored.result.F is not None and stored.result.F.shape[0] > 0
    assert stored.manifest.resolved_spec["seed"] == 1
    assert stored.manifest["provenance"]["entry_point"]["kind"] == "cli"
