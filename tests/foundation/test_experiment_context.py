"""Tests for Experiment context manager."""

from __future__ import annotations

import pytest


def _has_pandas() -> bool:
    """Check if pandas is available."""
    try:
        import pandas  # noqa: F401
        return True
    except ImportError:
        return False


class TestExperimentBasics:
    """Test basic Experiment functionality."""

    @pytest.mark.smoke
    def test_experiment_context_manager(self):
        """Experiment should work as context manager."""
        from vamos import Experiment, ZDT1

        with Experiment("test", verbose=False) as exp:
            result = exp.optimize(ZDT1(n_var=10), "nsgaii", max_evaluations=500, pop_size=20)

        assert result is not None
        assert result.F.shape[0] > 0
        assert len(exp.runs) == 1

    @pytest.mark.smoke
    def test_experiment_multiple_runs(self):
        """Experiment should track multiple runs."""
        from vamos import Experiment, ZDT1

        with Experiment("test", verbose=False) as exp:
            exp.optimize(ZDT1(n_var=10), "nsgaii", max_evaluations=500, pop_size=20)
            exp.optimize(ZDT1(n_var=10), "spea2", max_evaluations=500, pop_size=20)

        assert len(exp.runs) == 2
        assert len(exp.results) == 2

    def test_experiment_run_records(self):
        """Run records should contain metadata."""
        from vamos import Experiment, ZDT1

        with Experiment("test", verbose=False) as exp:
            exp.optimize(ZDT1(n_var=10), "nsgaii", max_evaluations=500, pop_size=20)

        run = exp.runs[0]
        assert run.algorithm == "nsgaii"
        assert run.n_solutions > 0
        assert run.elapsed_seconds > 0
        assert run.metadata["max_evaluations"] == 500


class TestExperimentSummary:
    """Test experiment summary."""

    def test_summary_basic(self):
        """summary() should return ExperimentSummary."""
        from vamos import Experiment, ZDT1

        with Experiment("test", verbose=False) as exp:
            exp.optimize(ZDT1(n_var=10), "nsgaii", max_evaluations=500, pop_size=20)

        summary = exp.summary()
        assert summary.name == "test"
        assert summary.total_runs == 1
        assert summary.total_time_seconds > 0

    def test_summary_str(self):
        """Summary should have nice string representation."""
        from vamos import Experiment, ZDT1

        with Experiment("test", verbose=False) as exp:
            exp.optimize(ZDT1(n_var=10), "nsgaii", max_evaluations=500, pop_size=20)

        summary = exp.summary()
        summary_str = str(summary)
        assert "test" in summary_str
        assert "nsgaii" in summary_str


class TestExperimentResults:
    """Test experiment result access."""

    def test_results_property(self):
        """results property should return OptimizationResult list."""
        from vamos import Experiment, OptimizationResult, ZDT1

        with Experiment("test", verbose=False) as exp:
            exp.optimize(ZDT1(n_var=10), "nsgaii", max_evaluations=500, pop_size=20)

        results = exp.results
        assert len(results) == 1
        assert isinstance(results[0], OptimizationResult)

    def test_best_run(self):
        """best_run() should return best by metric."""
        from vamos import Experiment, ZDT1

        with Experiment("test", verbose=False) as exp:
            exp.optimize(ZDT1(n_var=10), "nsgaii", max_evaluations=500, pop_size=20)
            exp.optimize(ZDT1(n_var=10), "nsgaii", max_evaluations=1000, pop_size=20)

        best = exp.best_run("time")
        assert best is not None
        # First run should be faster (fewer evals)
        assert best.metadata["max_evaluations"] == 500

    def test_compare(self):
        """compare() should return comparison dict."""
        from vamos import Experiment, ZDT1

        with Experiment("test", verbose=False) as exp:
            exp.optimize(ZDT1(n_var=10), "nsgaii", max_evaluations=500, pop_size=20)

        comparison = exp.compare()
        assert len(comparison) == 1
        key = list(comparison.keys())[0]
        assert "algorithm" in comparison[key]
        assert "n_solutions" in comparison[key]


class TestExperimentDataFrame:
    """Test DataFrame conversion."""

    @pytest.mark.skipif(not _has_pandas(), reason="pandas not installed")
    def test_to_dataframe(self):
        """to_dataframe() should create valid DataFrame."""
        from vamos import Experiment, ZDT1

        with Experiment("test", verbose=False) as exp:
            exp.optimize(ZDT1(n_var=10), "nsgaii", max_evaluations=500, pop_size=20)
            exp.optimize(ZDT1(n_var=10), "spea2", max_evaluations=500, pop_size=20)

        df = exp.to_dataframe()
        assert len(df) == 2
        assert "algorithm" in df.columns
        assert "n_solutions" in df.columns
        assert "time_seconds" in df.columns


class TestExperimentSave:
    """Test saving results."""

    def test_save_to_output_dir(self, tmp_path):
        """Results should be saved to output_dir."""
        from vamos import Experiment, ZDT1

        with Experiment("test", output_dir=tmp_path, verbose=False) as exp:
            exp.optimize(ZDT1(n_var=10), "nsgaii", max_evaluations=500, pop_size=20, save=True)

        # Check that output was created
        output_path = tmp_path / "test"
        assert output_path.exists()


class TestExperimentFunction:
    """Test functional experiment() context manager."""

    @pytest.mark.smoke
    def test_experiment_function(self):
        """experiment() function should work like Experiment class."""
        from vamos import ZDT1
        from vamos.experiment import experiment

        with experiment("test", verbose=False) as exp:
            result = exp.optimize(ZDT1(n_var=10), "nsgaii", max_evaluations=500, pop_size=20)

        assert result is not None
        assert len(exp.runs) == 1


class TestExperimentErrors:
    """Test error handling."""

    def test_optimize_outside_context(self):
        """optimize() should raise if not in context."""
        from vamos import Experiment, ZDT1

        exp = Experiment("test", verbose=False)
        with pytest.raises(RuntimeError, match="not active"):
            exp.optimize(ZDT1(n_var=10), "nsgaii", max_evaluations=100)

    def test_best_run_invalid_metric(self):
        """best_run() with invalid metric should raise."""
        from vamos import Experiment, ZDT1

        with Experiment("test", verbose=False) as exp:
            exp.optimize(ZDT1(n_var=10), "nsgaii", max_evaluations=500, pop_size=20)

        with pytest.raises(ValueError, match="Unknown metric"):
            exp.best_run("invalid")


class TestExperimentSeeding:
    """Test seed handling."""

    def test_auto_increment_seed(self):
        """Seeds should auto-increment for reproducibility."""
        from vamos import Experiment, ZDT1

        with Experiment("test", seed=100, verbose=False) as exp:
            exp.optimize(ZDT1(n_var=10), "nsgaii", max_evaluations=500, pop_size=20)
            exp.optimize(ZDT1(n_var=10), "nsgaii", max_evaluations=500, pop_size=20)

        assert exp.runs[0].metadata["seed"] == 100
        assert exp.runs[1].metadata["seed"] == 101

    def test_custom_seed(self):
        """Custom seed should override auto-increment."""
        from vamos import Experiment, ZDT1

        with Experiment("test", seed=100, verbose=False) as exp:
            exp.optimize(ZDT1(n_var=10), "nsgaii", max_evaluations=500, pop_size=20, seed=999)

        assert exp.runs[0].metadata["seed"] == 999


class TestExperimentImports:
    """Test imports from vamos."""

    def test_all_exports_available(self):
        """All experiment exports should be importable."""
        from vamos import Experiment, experiment, RunRecord, ExperimentSummary

        assert Experiment is not None
        assert experiment is not None
        assert RunRecord is not None
        assert ExperimentSummary is not None
