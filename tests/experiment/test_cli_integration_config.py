import json
import sys

import vamos
from vamos.ux.visualization import plotting


def test_cli_with_config_file_creates_artifacts(monkeypatch, tmp_path):
    output_root = tmp_path / "results"
    config_file = tmp_path / "spec.json"
    # Minimal config file exercising algorithm/engine/output_root overrides
    spec = {
        "version": "1",
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
    monkeypatch.setattr(plotting, "plot_pareto_front", lambda *args, **kwargs: None)

    argv = ["prog", "--config", str(config_file)]
    monkeypatch.setattr(sys, "argv", argv)

    from vamos.experiment.cli.main import main

    main()

    run_dir = output_root / "ZDT1" / "nsgaii" / "numpy" / "seed_5"
    assert {path.name for path in run_dir.iterdir()} == {"manifest.json", "result.npz", "environment.json"}
    stored = vamos.load_run(run_dir, verify="all")
    assert stored.result.F is not None and stored.result.F.shape[0] > 0
    assert stored.manifest.resolved_spec["algorithm"]["component_id"] == "vamos.algorithm:nsgaii@1"
    assert stored.manifest.resolved_spec["backend"]["kernel"]["component_id"] == "vamos.kernel:numpy@1"
    assert stored.manifest.resolved_spec["problem"]["component_id"] == "vamos.problem:zdt1@1"
    assert stored.manifest.resolved_spec["seed"] == 5


def test_cli_runs_spea2_from_config(monkeypatch, tmp_path):
    output_root = tmp_path / "results"
    config_file = tmp_path / "spea2_spec.json"
    spec = {
        "version": "1",
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
    monkeypatch.setattr(plotting, "plot_pareto_front", lambda *args, **kwargs: None)

    argv = ["prog", "--config", str(config_file)]
    monkeypatch.setattr(sys, "argv", argv)

    from vamos.experiment.cli.main import main

    main()

    run_dir = output_root / "ZDT1" / "spea2" / "numpy" / "seed_2"
    stored = vamos.load_run(run_dir, verify="all")
    assert stored.result.F is not None
    assert stored.result.data["archive"]["F"] is not None


def test_cli_config_null_seed_is_resolved_before_execution(monkeypatch, tmp_path):
    output_root = tmp_path / "results"
    config_file = tmp_path / "null-seed.json"
    config_file.write_text(
        json.dumps(
            {
                "version": "1",
                "defaults": {
                    "problem": "zdt1",
                    "algorithm": "nsgaii",
                    "engine": "numpy",
                    "population_size": 4,
                    "max_evaluations": 4,
                    "output_root": str(output_root),
                    "seed": None,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["prog", "--config", str(config_file)])

    from vamos.experiment.cli.main import main

    main()

    manifests = list(output_root.rglob("manifest.json"))
    assert len(manifests) == 1
    stored = vamos.load_run(manifests[0].parent)
    assert stored.manifest.requested_spec["defaults"]["seed"] is None
    assert isinstance(stored.manifest.resolved_spec["seed"], int)
