import sys
from types import SimpleNamespace

import numpy as np
import pytest

from vamos.experiment import cli
from vamos.foundation.core import runner
from vamos.foundation.problem.tsp import TSPProblem
from vamos.experiment.study.runner import StudyRunner, StudyTask


def test_cli_hv_threshold_requires_reference_for_non_zdt(monkeypatch):
    default_cfg = runner.ExperimentConfig()
    argv = ["prog", "--problem", "dtlz1", "--hv-threshold", "0.5"]
    monkeypatch.setenv("PYTHONHASHSEED", "0")  # keep argparse deterministic
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        cli.parse_args(default_cfg)


def test_cli_hv_threshold_uses_builtin_reference_for_zdt1(monkeypatch):
    default_cfg = runner.ExperimentConfig()
    argv = ["prog", "--problem", "zdt1", "--hv-threshold", "0.25"]
    monkeypatch.setattr(sys, "argv", argv)
    args = cli.parse_args(default_cfg)
    assert args.hv_reference_front
    assert "ZDT1" in args.hv_reference_front.upper()


def test_build_hv_stop_config_uses_builtin_front():
    cfg = runner.build_hv_stop_config(0.1, None, "zdt1")
    assert cfg["target_value"] > 0.0
    assert len(cfg["reference_point"]) == 2
    assert cfg["reference_front_path"].upper().endswith("ZDT1.CSV")


def test_permutation_problem_requires_nsgaii():
    selection = SimpleNamespace(
        instantiate=lambda: TSPProblem(n_cities=6), spec=SimpleNamespace(key="tsp6"), n_var=6, n_obj=2
    )
    config = runner.ExperimentConfig(population_size=4, offspring_population_size=4, max_evaluations=8, seed=1)
    with pytest.raises(ValueError):
        runner.run_single(
            "numpy",
            "moead",
            selection,
            config,
        )


def test_study_runner_mirrors_outputs(monkeypatch, tmp_path):
    base_root = tmp_path / "results"
    mirror_root = tmp_path / "mirror"
    base_root.mkdir()
    mirror_root.mkdir()
    monkeypatch.setenv("VAMOS_OUTPUT_ROOT", str(base_root))

    def fake_run_single(engine_name, algorithm_name, selection, config, **kwargs):
        out_dir = base_root / selection.spec.key.upper() / algorithm_name / engine_name / f"seed_{config.seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in ("FUN.csv", "time.txt", "metadata.json"):
            (out_dir / name).write_text("dummy", encoding="utf-8")
        return {
            "engine": engine_name,
            "algorithm": algorithm_name,
            "time_ms": 1.0,
            "evaluations": 2,
            "evals_per_sec": 2.0,
            "F": np.array([[1.0, 1.0]]),
            "output_dir": str(out_dir),
        }

    # Import module explicitly to avoid shadowing by vamos.experiment function
    from vamos.experiment.study import runner as study_runner_module
    monkeypatch.setattr(study_runner_module, "run_single", fake_run_single)

    tasks = [
        StudyTask(
            algorithm="nsgaii",
            engine="numpy",
            problem="zdt1",
            n_var=6,
            seed=1,
        )
    ]
    runner_obj = StudyRunner(verbose=False, mirror_output_roots=[mirror_root])
    results = runner_obj.run(tasks)

    assert results, "No study results returned"
    mirrored_fun = mirror_root / "ZDT1" / "nsgaii" / "numpy" / "seed_1" / "FUN.csv"
    assert mirrored_fun.exists(), "FUN.csv was not mirrored to the target root"
