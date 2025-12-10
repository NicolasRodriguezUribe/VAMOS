import pytest
import numpy as np
import pandas as pd
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from vamos.analysis.tuning_viz import (
    tuning_result_to_dataframe,
    plot_tuning_scatter,
    plot_objective_tradeoff,
    plot_reduced_front,
    study_results_to_dataframe,
    summarize_by_algorithm,
)


class _FakeTuningResult:
    def __init__(self):
        self.unit_vectors = np.array([[0.1, 0.2], [0.5, 0.8]])
        self.objectives = np.array([[1.0, 2.0], [0.5, 3.0]])
        self.assignments = [{"a": 1}, {"a": 2}]


def test_tuning_result_to_dataframe():
    res = _FakeTuningResult()
    df = tuning_result_to_dataframe(res, param_names=["p1", "p2"])
    assert isinstance(df, pd.DataFrame)
    assert {"p1", "p2", "obj_0", "obj_1"}.issubset(df.columns)


def test_plot_helpers_do_not_crash():
    res = _FakeTuningResult()
    df = tuning_result_to_dataframe(res, param_names=["p1", "p2"])
    plot_tuning_scatter(df, "p1", "p2", color_by="obj_0")
    plot_objective_tradeoff(df, "obj_0", "obj_1")
    F = np.random.rand(10, 3)
    plot_reduced_front(F, target_dim=2)


def test_study_df_and_summary():
    class _FakeSelection:
        def __init__(self, key):
            self.spec = type("spec", (), {"key": key})

    class _FakeResult:
        def __init__(self, key):
            self.selection = _FakeSelection(key)
            self.metrics = {
                "algorithm": "nsgaii",
                "engine": "numpy",
                "seed": 1,
                "hv": 0.5,
                "time_ms": 10.0,
                "evaluations": 20,
                "spread": 1.0,
            }

    study_results = [_FakeResult("zdt1"), _FakeResult("zdt2")]
    df = study_results_to_dataframe(study_results)
    assert {"problem", "algorithm", "hv"}.issubset(df.columns)
    summary = summarize_by_algorithm(df)
    assert "hv_mean" in summary.columns
