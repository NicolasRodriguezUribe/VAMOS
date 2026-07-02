import json
import sys

import pytest

from vamos.foundation.exceptions import VAMOSError


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


def test_cli_vamos_error_barrier_suppresses_traceback(monkeypatch, capsys):
    from vamos.experiment.cli import main as cli_main

    def boom(argv=None):  # noqa: ANN001
        raise VAMOSError("bad config", suggestion="fix it")

    monkeypatch.setattr(sys, "argv", ["prog"])
    monkeypatch.setattr(cli_main, "_dispatch_subcommand", lambda argv: False)
    monkeypatch.setattr(cli_main, "_config_only_path", lambda argv: None)
    monkeypatch.setattr(cli_main, "_run_standard_cli", boom)

    with pytest.raises(SystemExit) as excinfo:
        cli_main.main()

    captured = capsys.readouterr()
    assert excinfo.value.code == 2
    assert "Error: bad config" in captured.err
    assert "Suggestion: fix it" in captured.err
    assert "Traceback" not in captured.err


def test_cli_traceback_flag_preserves_vamos_error(monkeypatch):
    from vamos.experiment.cli import main as cli_main

    def boom(argv=None):  # noqa: ANN001
        raise VAMOSError("bad config")

    monkeypatch.setattr(sys, "argv", ["prog", "--traceback"])
    monkeypatch.setattr(cli_main, "_dispatch_subcommand", lambda argv: False)
    monkeypatch.setattr(cli_main, "_config_only_path", lambda argv: None)
    monkeypatch.setattr(cli_main, "_run_standard_cli", boom)

    with pytest.raises(VAMOSError):
        cli_main.main()
