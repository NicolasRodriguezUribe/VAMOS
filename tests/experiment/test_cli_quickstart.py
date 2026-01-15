import json

import pytest

from vamos.engine.config.spec import EXPERIMENT_SPEC_VERSION
from vamos.experiment.cli import quickstart


def test_quickstart_template_demo_exists() -> None:
    template = quickstart.get_template("demo")
    assert template.defaults.problem == "zdt1"


def test_quickstart_template_list_smoke(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        quickstart.run_quickstart(["--template", "list", "--yes"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "Quickstart templates:" in out
    assert "demo" in out


def test_quickstart_write_spec(tmp_path) -> None:
    template = quickstart.get_template("demo")
    spec_path = quickstart._write_spec(
        title="Quickstart: Demo",
        problem=template.defaults.problem,
        algorithm=template.defaults.algorithm,
        engine=template.defaults.engine,
        budget=template.defaults.budget,
        pop_size=template.defaults.pop_size,
        seed=template.defaults.seed,
        output_root=str(tmp_path / "results"),
        plot=template.defaults.plot,
        config_path=str(tmp_path / "quickstart.json"),
    )
    data = json.loads(spec_path.read_text(encoding="utf-8"))
    assert data["version"] == EXPERIMENT_SPEC_VERSION
    defaults = data["defaults"]
    assert defaults["problem"] == "zdt1"
    assert defaults["algorithm"] == "nsgaii"
    assert defaults["engine"] == "numpy"
