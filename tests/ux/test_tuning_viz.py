import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from vamos.ux.analysis.tuning_viz import (
    plot_objective_tradeoff,
    plot_reduced_front,
    plot_tuning_scatter,
    study_summary_to_dataframe,
    summarize_by_algorithm,
    tuning_result_to_dataframe,
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
    F = np.random.default_rng(0).random((10, 3))
    plot_reduced_front(F, target_dim=2)


def test_study_df_and_summary(tmp_path):
    from vamos import StudySpec, create_study

    completed = create_study(
        StudySpec(
            problems=["zdt1", "zdt2"],
            algorithms=["nsgaii"],
            seeds=[1],
            max_evaluations=4,
            pop_size=4,
            engine="numpy",
        ),
        output=tmp_path / "study",
    ).run()
    df = study_summary_to_dataframe(completed.summarize())
    assert {"study_id", "task_id", "selected_run_id", "problem", "algorithm"}.issubset(df.columns)
    assert set(df["problem"]) == {"zdt1", "zdt2"}
    summary = summarize_by_algorithm(df)
    assert "runtime_ms_mean" in summary.columns
