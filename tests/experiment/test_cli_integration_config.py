import json
import sys

import numpy as np

from vamos.experiment import runner as experiment_runner


def test_cli_with_config_file_creates_artifacts(monkeypatch, tmp_path):
    output_root = tmp_path / "results"
    config_file = tmp_path / "spec.json"
    # Minimal config file exercising algorithm/engine/output_root overrides
    spec = {
        "defaults": {
            "algorithm": "nsgaii",
            "engine": "numpy",
            "population_size": 10,
            "max_evaluations": 30,
            "output_root": str(output_root),
            "nsgaii": {"crossover": {"method": "sbx", "prob": 0.9, "eta": 15.0}},
        },
        "problems": {"zdt1": {"seed": 5}},
    }
    config_file.write_text(json.dumps(spec), encoding="utf-8")

    monkeypatch.setenv("VAMOS_OUTPUT_ROOT", str(output_root))
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    # Avoid plotting during tests
    monkeypatch.setattr(experiment_runner.plotting, "plot_pareto_front", lambda *args, **kwargs: None)

    argv = ["prog", "--config", str(config_file)]
    monkeypatch.setattr(sys, "argv", argv)

    from vamos.experiment.cli.main import main

    main()

    run_dir = output_root / "ZDT1" / "nsgaii" / "numpy" / "seed_5"
    fun_path = run_dir / "FUN.csv"
    meta_path = run_dir / "metadata.json"
    resolved_path = run_dir / "resolved_config.json"

    assert fun_path.exists(), "FUN.csv not created"
    assert meta_path.exists(), "metadata.json not created"
    assert resolved_path.exists(), "resolved_config.json not created"

    fun = np.loadtxt(fun_path, delimiter=",")
    assert fun.shape[0] > 0

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    assert meta["algorithm"] == "nsgaii"
    assert meta["backend"] == "numpy"
    assert meta["seed"] == 5
    assert meta["problem"]["key"] == "zdt1"
    assert resolved["algorithm"] == "nsgaii"
    assert resolved["engine"] == "numpy"
    assert resolved["problem"] == "zdt1"
    assert resolved["seed"] == 5


def test_cli_runs_spea2_from_config(monkeypatch, tmp_path):
    output_root = tmp_path / "results"
    config_file = tmp_path / "spea2_spec.json"
    spec = {
        "defaults": {
            "algorithm": "spea2",
            "engine": "numpy",
            "population_size": 8,
            "max_evaluations": 16,
            "output_root": str(output_root),
            "spea2": {"crossover": {"method": "sbx", "prob": 0.9, "eta": 15.0}},
        },
        "problems": {"zdt1": {"seed": 2}},
    }
    config_file.write_text(json.dumps(spec), encoding="utf-8")

    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setattr(experiment_runner.plotting, "plot_pareto_front", lambda *args, **kwargs: None)

    argv = ["prog", "--config", str(config_file)]
    monkeypatch.setattr(sys, "argv", argv)

    from vamos.experiment.cli.main import main

    main()

    run_dir = output_root / "ZDT1" / "spea2" / "numpy" / "seed_2"
    fun_path = run_dir / "FUN.csv"
    archive_path = run_dir / "ARCHIVE_FUN.csv"

    assert fun_path.exists()
    assert archive_path.exists()
