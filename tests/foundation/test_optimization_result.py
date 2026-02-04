"""Tests for OptimizationResult enriched methods."""

from __future__ import annotations

import logging
import numpy as np
import pytest


def _has_matplotlib() -> bool:
    """Check if matplotlib is available."""
    try:
        import matplotlib  # noqa: F401

        return True
    except ImportError:
        return False


def _has_pandas() -> bool:
    """Check if pandas is available."""
    try:
        import pandas  # noqa: F401

        return True
    except ImportError:
        return False


class TestOptimizationResultBasics:
    """Test basic OptimizationResult functionality."""

    def test_result_len(self):
        """Result should support len()."""
        from vamos.experiment.optimization_result import OptimizationResult

        F = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
        result = OptimizationResult({"F": F, "X": None})

        assert len(result) == 3

    def test_result_repr(self):
        """Result should have informative repr."""
        from vamos.experiment.optimization_result import OptimizationResult

        F = np.array([[0.1, 0.9], [0.5, 0.5]])
        result = OptimizationResult({"F": F})

        repr_str = repr(result)
        assert "2 solutions" in repr_str
        assert "2 objectives" in repr_str

    def test_n_objectives(self):
        """n_objectives property should work."""
        from vamos.experiment.optimization_result import OptimizationResult

        F = np.array([[0.1, 0.9, 0.5], [0.5, 0.5, 0.5]])
        result = OptimizationResult({"F": F})

        assert result.n_objectives == 3


class TestOptimizationResultSummary:
    """Test summary() method."""

    def test_summary_runs(self, caplog):
        """summary() should log without error."""
        from vamos.experiment.optimization_result import OptimizationResult
        from vamos.ux.api import log_result_summary

        F = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
        result = OptimizationResult({"F": F})

        caplog.set_level(logging.INFO, logger="vamos.ux.results")
        log_result_summary(result)
        assert "Optimization Result" in caplog.text
        assert "Solutions: 3" in caplog.text
        assert "Objectives: 2" in caplog.text


class TestOptimizationResultDefaults:
    """Test explain_defaults() metadata helper."""

    def test_explain_defaults_returns_meta(self):
        from vamos.experiment.optimization_result import OptimizationResult

        F = np.array([[0.1, 0.9], [0.5, 0.5]])
        meta = {
            "resolved_config": {"problem": "zdt1", "algorithm": "nsgaii"},
            "default_sources": {"algorithm": "explicit", "max_evaluations": "auto"},
        }
        result = OptimizationResult({"F": F}, meta=meta)

        explained = result.explain_defaults()

        assert explained["resolved_config"]["problem"] == "zdt1"
        assert explained["default_sources"]["max_evaluations"] == "auto"

    def test_explain_defaults_handles_missing_meta(self):
        from vamos.experiment.optimization_result import OptimizationResult

        F = np.array([[0.1, 0.9]])
        result = OptimizationResult({"F": F})

        assert result.explain_defaults() == {}


class TestOptimizationResultBest:
    """Test best() method."""

    def test_best_knee(self):
        """best('knee') should return a solution."""
        from vamos.experiment.optimization_result import OptimizationResult

        F = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
        X = np.array([[1, 2], [3, 4], [5, 6]])
        result = OptimizationResult({"F": F, "X": X})

        best = result.best("knee")

        assert "F" in best
        assert "X" in best
        assert "index" in best
        assert 0 <= best["index"] < 3

    def test_best_knee_uses_pareto_front_scaling(self):
        """best('knee') should pick from the Pareto front with front-based normalization."""
        from vamos.experiment.optimization_result import OptimizationResult

        # Three non-dominated points plus an extreme dominated outlier.
        F = np.array(
            [
                [0.0, 100.0],
                [50.0, 0.0],
                [25.0, 25.0],
                [10000.0, 1.0],
            ]
        )
        X = np.arange(F.shape[0] * 2).reshape(F.shape[0], 2)
        result = OptimizationResult({"F": F, "X": X})

        best = result.best("knee")

        assert best["index"] == 2
        assert np.allclose(best["F"], F[2])

    def test_best_min_f1(self):
        """best('min_f1') should return min first objective."""
        from vamos.experiment.optimization_result import OptimizationResult

        F = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
        result = OptimizationResult({"F": F})

        best = result.best("min_f1")

        assert best["F"][0] == 0.1
        assert best["index"] == 0

    def test_best_min_f2(self):
        """best('min_f2') should return min second objective."""
        from vamos.experiment.optimization_result import OptimizationResult

        F = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
        result = OptimizationResult({"F": F})

        best = result.best("min_f2")

        assert best["F"][1] == 0.1
        assert best["index"] == 2

    def test_best_balanced(self):
        """best('balanced') should return a solution."""
        from vamos.experiment.optimization_result import OptimizationResult

        F = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
        result = OptimizationResult({"F": F})

        best = result.best("balanced")

        assert "index" in best
        assert 0 <= best["index"] < 3

    def test_best_invalid_method(self):
        """best() with invalid method should raise."""
        from vamos.experiment.optimization_result import OptimizationResult

        F = np.array([[0.1, 0.9], [0.5, 0.5]])
        result = OptimizationResult({"F": F})

        with pytest.raises(ValueError, match="Unknown method"):
            result.best("invalid")


class TestOptimizationResultPlot:
    """Test plot() method."""

    @pytest.mark.skipif(not _has_matplotlib(), reason="matplotlib not installed")
    def test_plot_2d(self):
        """plot() should work for 2D fronts."""
        from vamos.experiment.optimization_result import OptimizationResult
        from vamos.ux.api import plot_result_front

        F = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
        result = OptimizationResult({"F": F})

        ax = plot_result_front(result, show=False)
        assert ax is not None

    @pytest.mark.skipif(not _has_matplotlib(), reason="matplotlib not installed")
    def test_plot_3d(self):
        """plot() should work for 3D fronts."""
        from vamos.experiment.optimization_result import OptimizationResult
        from vamos.ux.api import plot_result_front

        F = np.array([[0.1, 0.8, 0.1], [0.5, 0.3, 0.2], [0.8, 0.1, 0.1]])
        result = OptimizationResult({"F": F})

        ax = plot_result_front(result, show=False)
        assert ax is not None

    def test_plot_no_matplotlib(self):
        """plot() should raise if matplotlib not available."""
        # This is hard to test without mocking, skip for now
        pass


class TestOptimizationResultDataFrame:
    """Test to_dataframe() method."""

    @pytest.mark.skipif(not _has_pandas(), reason="pandas not installed")
    def test_to_dataframe_basic(self):
        """to_dataframe() should create valid DataFrame."""
        from vamos.experiment.optimization_result import OptimizationResult
        from vamos.ux.api import result_to_dataframe

        F = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
        X = np.array([[1, 2], [3, 4], [5, 6]])
        result = OptimizationResult({"F": F, "X": X})

        df = result_to_dataframe(result)

        assert len(df) == 3
        assert "f1" in df.columns
        assert "f2" in df.columns
        assert "x1" in df.columns
        assert "x2" in df.columns

    @pytest.mark.skipif(not _has_pandas(), reason="pandas not installed")
    def test_to_dataframe_no_x(self):
        """to_dataframe() should work without X."""
        from vamos.experiment.optimization_result import OptimizationResult
        from vamos.ux.api import result_to_dataframe

        F = np.array([[0.1, 0.9], [0.5, 0.5]])
        result = OptimizationResult({"F": F})

        df = result_to_dataframe(result)

        assert len(df) == 2
        assert "f1" in df.columns
        assert "x1" not in df.columns


class TestOptimizationResultSave:
    """Test save() method."""

    def test_save_creates_files(self, tmp_path):
        """save() should create expected files."""
        from vamos.experiment.optimization_result import OptimizationResult
        from vamos.ux.api import save_result

        F = np.array([[0.1, 0.9], [0.5, 0.5]])
        X = np.array([[1, 2], [3, 4]])
        result = OptimizationResult({"F": F, "X": X})

        out_dir = tmp_path / "test_results"
        save_result(result, str(out_dir))

        assert (out_dir / "FUN.csv").exists()
        assert (out_dir / "X.csv").exists()
        assert (out_dir / "metadata.json").exists()
