import json
import sys
from pathlib import Path

import pytest


def test_cli_ablation_runs_and_writes_variants(monkeypatch, tmp_path):
    output_root = tmp_path / "ablation_results"
    config = {
        "algorithm": "moead",
        "engine": "numpy",
        "output_root": str(output_root),
        "default_max_evals": 12,
        "problems": ["zdt1"],
        "seeds": [1],
        "base_config": {"population_size": 6, "offspring_population_size": 6},
        "variants": [
            {"name": "baseline"},
            {
                "name": "tuned",
                "config_overrides": {"population_size": 8, "offspring_population_size": 8},
                "moead_variation": {"aggregation": {"method": "pbi", "theta": 5.0}},
            },
        ],
    }
    config_path = tmp_path / "ablation.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    argv = ["prog", "ablation", "--config", str(config_path)]
    monkeypatch.setattr(sys, "argv", argv)

    from vamos.experiment.cli.main import main

    main()

    baseline_dir = output_root / "baseline" / "ZDT1" / "moead" / "numpy" / "seed_1"
    tuned_dir = output_root / "tuned" / "ZDT1" / "moead" / "numpy" / "seed_1"
    assert (baseline_dir / "FUN.csv").exists()
    assert (tuned_dir / "FUN.csv").exists()

    summary_path = output_root / "summary" / "ablation_metrics.csv"
    assert summary_path.exists()


def test_cli_ablation_invalid_config_raises(monkeypatch, tmp_path):
    config = {
        "algorithm": "nsgaii",
        "engine": "numpy",
        "default_max_evals": 10,
        "problems": ["zdt1"],
        "seeds": [1],
    }
    config_path = tmp_path / "ablation_invalid.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    argv = ["prog", "ablation", "--config", str(config_path)]
    monkeypatch.setattr(sys, "argv", argv)

    from vamos.experiment.cli.main import main

    with pytest.raises(ValueError):
        main()


def test_cli_ablation_schema_validation(monkeypatch, tmp_path):
    config = {
        "algorithm": "nsgaii",
        "engine": "numpy",
        "default_max_evals": 10,
        "problems": "zdt1",
        "seeds": [1],
        "variants": [{"name": "baseline"}],
    }
    config_path = tmp_path / "ablation_invalid_schema.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    argv = ["prog", "ablation", "--config", str(config_path)]
    monkeypatch.setattr(sys, "argv", argv)

    from vamos.experiment.cli.main import main

    with pytest.raises(ValueError):
        main()


def test_cli_ablation_rejects_unknown_keys(monkeypatch, tmp_path):
    config = {
        "algorithm": "nsgaii",
        "engine": "numpy",
        "default_max_evals": 10,
        "problems": ["zdt1"],
        "seeds": [1],
        "variants": [{"name": "baseline"}],
        "unexpected": 123,
    }
    config_path = tmp_path / "ablation_unknown.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    argv = ["prog", "ablation", "--config", str(config_path)]
    monkeypatch.setattr(sys, "argv", argv)

    from vamos.experiment.cli.main import main

    with pytest.raises(ValueError):
        main()


def test_cli_ablation_rejects_unknown_variation_keys(monkeypatch, tmp_path):
    config = {
        "algorithm": "moead",
        "engine": "numpy",
        "default_max_evals": 10,
        "problems": ["zdt1"],
        "seeds": [1],
        "variants": [
            {"name": "baseline", "moead_variation": {"unknown_key": 1}},
        ],
    }
    config_path = tmp_path / "ablation_bad_variation.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    argv = ["prog", "ablation", "--config", str(config_path)]
    monkeypatch.setattr(sys, "argv", argv)

    from vamos.experiment.cli.main import main

    with pytest.raises(ValueError):
        main()
