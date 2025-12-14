from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from vamos.foundation.core.metadata import build_run_metadata


class DummyProblem:
    def __init__(self, n_var: int = 2, n_obj: int = 2, encoding: str = "real"):
        self.n_var = n_var
        self.n_obj = n_obj
        self.encoding = encoding


class DummySpec:
    def __init__(self, label: str, key: str, description: str | None = None):
        self.label = label
        self.key = key
        self.description = description


class DummySelection:
    def __init__(self, spec: DummySpec):
        self.spec = spec
        self.n_var = 3
        self.n_obj = 2

    def instantiate(self):
        return DummyProblem(n_var=self.n_var, n_obj=self.n_obj, encoding="real")


class DummyKernel:
    def capabilities(self):
        return ["hypervolume"]

    def device(self):
        return "cpu"


class DummyConfig:
    def __init__(self):
        self.crossover = ("sbx", {"prob": 0.9})
        self.mutation = ("pm", {"prob": 0.1})
        self.repair = ("clip", {})

    def to_dict(self):
        return {"pop_size": 50}


class DummyExperimentConfig(SimpleNamespace):
    pass


def test_build_run_metadata_populates_core_fields(tmp_path: Path):
    selection = DummySelection(DummySpec("Test Problem", "test_key", "desc"))
    cfg_data = DummyConfig()
    metrics = {
        "time_ms": 123.4,
        "evaluations": 200,
        "evals_per_sec": 10.0,
        "spread": 0.5,
    }
    kernel = DummyKernel()
    config = DummyExperimentConfig(title="Run", population_size=50, max_evaluations=200, output_root=str(tmp_path))
    metadata = build_run_metadata(
        selection,
        "nsgaii",
        "numpy",
        cfg_data,
        metrics,
        kernel_backend=kernel,
        seed=42,
        config=config,
        project_root=tmp_path,
    )

    assert metadata["algorithm"] == "nsgaii"
    assert metadata["backend"] == "numpy"
    assert metadata["seed"] == 42
    assert metadata["problem"]["label"] == "Test Problem"
    assert metadata["problem"]["encoding"] == "real"
    assert metadata["metrics"]["evaluations"] == 200
    assert metadata["operators"]["crossover"]["name"] == "sbx"
    assert metadata["backend_info"]["device"] == "cpu"
