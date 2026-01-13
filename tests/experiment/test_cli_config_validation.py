import json
import sys

import pytest


def test_cli_validate_config_exits_cleanly(monkeypatch, tmp_path):
    config_file = tmp_path / "spec.json"
    config_file.write_text(
        json.dumps(
            {
                "version": "1",
                "defaults": {
                    "algorithm": "nsgaii",
                    "engine": "numpy",
                    "population_size": 10,
                    "max_evaluations": 30,
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["prog", "--config", str(config_file), "--validate-config"])
    from vamos.experiment.cli.main import main

    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0


def test_cli_config_requires_version(monkeypatch, tmp_path):
    config_file = tmp_path / "spec.json"
    config_file.write_text(
        json.dumps(
            {
                "defaults": {
                    "algorithm": "nsgaii",
                    "engine": "numpy",
                    "population_size": 10,
                    "max_evaluations": 30,
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["prog", "--config", str(config_file)])
    from vamos.experiment.cli.main import main

    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 2
