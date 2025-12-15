"""Tests for vamos.quick API - one-liner experiment convenience functions."""

from __future__ import annotations

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


class TestQuickAPIImports:
    """Test that quick API is correctly exported from vamos root."""

    def test_import_quick_functions(self):
        """All quick functions should be importable from vamos."""
        from vamos import (
            QuickResult,
            run,
            run_moead,
            run_nsgaii,
            run_nsga3,
            run_smsemoa,
            run_spea2,
        )

        assert callable(run)
        assert callable(run_nsgaii)
        assert callable(run_moead)
        assert callable(run_spea2)
        assert callable(run_smsemoa)
        assert callable(run_nsga3)
        assert QuickResult is not None


class TestQuickResultBasics:
    """Test QuickResult class functionality."""

    def test_quick_result_from_nsgaii(self):
        """QuickResult should have expected attributes after run."""
        from vamos import run_nsgaii

        result = run_nsgaii("zdt1", max_evaluations=100, pop_size=10, seed=42)

        assert len(result) > 0  # has solutions
        assert result.F.ndim == 2
        assert result.F.shape[1] == 2  # ZDT1 has 2 objectives
        assert np.isfinite(result.F).all()

    def test_quick_result_repr(self):
        """QuickResult repr should be informative."""
        from vamos import run_nsgaii

        result = run_nsgaii("zdt1", max_evaluations=100, pop_size=10)
        rep = repr(result)

        assert "QuickResult" in rep
        assert "nsgaii" in rep
        assert "solutions" in rep
        assert "objectives" in rep

    def test_quick_result_to_dict(self):
        """QuickResult should serialize to dict via best()."""
        from vamos import run_nsgaii

        result = run_nsgaii("zdt1", max_evaluations=100, pop_size=10)
        best = result.best("knee")  # Returns dict

        assert "F" in best
        assert "index" in best
        assert isinstance(best["index"], int)

    def test_quick_result_best(self):
        """Best solutions should be returned correctly."""
        from vamos import run_nsgaii

        result = run_nsgaii("zdt1", max_evaluations=100, pop_size=10)

        # Best using different methods
        knee = result.best("knee")
        assert "index" in knee
        assert 0 <= knee["index"] < len(result)

        min_f1 = result.best("min_f1")
        assert min_f1["F"][0] == result.F[:, 0].min()

        min_f2 = result.best("min_f2")
        assert min_f2["F"][1] == result.F[:, 1].min()

        balanced = result.best("balanced")
        assert "F" in balanced


@pytest.mark.slow
class TestQuickAPIAlgorithms:
    """Test all quick API algorithm functions."""

    def test_run_nsgaii(self):
        """run_nsgaii should complete and return valid results."""
        from vamos import run_nsgaii

        result = run_nsgaii(
            "zdt1",
            n_var=10,
            max_evaluations=200,
            pop_size=20,
            seed=42,
        )

        assert len(result) > 0
        assert np.isfinite(result.F).all()

    def test_run_moead(self):
        """run_moead should complete and return valid results."""
        from vamos import run_moead

        result = run_moead(
            "zdt1",
            n_var=10,
            max_evaluations=200,
            pop_size=20,
            seed=42,
        )

        assert len(result) > 0
        assert np.isfinite(result.F).all()

    def test_run_spea2(self):
        """run_spea2 should complete and return valid results."""
        from vamos import run_spea2

        result = run_spea2(
            "zdt1",
            n_var=10,
            max_evaluations=200,
            pop_size=20,
            seed=42,
        )

        assert len(result) > 0
        assert np.isfinite(result.F).all()

    def test_run_smsemoa(self):
        """run_smsemoa should complete and return valid results."""
        from vamos import run_smsemoa

        result = run_smsemoa(
            "zdt1",
            n_var=10,
            max_evaluations=200,
            pop_size=20,
            seed=42,
        )

        assert len(result) > 0
        assert np.isfinite(result.F).all()

    def test_run_nsga3_3obj(self):
        """run_nsga3 should work for 3-objective problems."""
        from vamos import run_nsga3

        result = run_nsga3(
            "dtlz2",
            n_var=10,
            n_obj=3,
            max_evaluations=200,
            pop_size=20,
            seed=42,
        )

        assert len(result) > 0
        assert result.F.shape[1] == 3  # 3 objectives
        assert np.isfinite(result.F).all()

    def test_run_generic_function(self):
        """run() should dispatch to correct algorithm."""
        from vamos import run

        # Test each algorithm via run()
        for algo in ["nsgaii", "moead", "spea2", "smsemoa"]:
            result = run(
                "zdt1",
                algo,
                max_evaluations=100,
                pop_size=10,
                seed=42,
            )
            assert len(result) > 0
            assert result.algorithm == algo


class TestQuickAPISmoke:
    """Fast smoke tests for quick API."""

    def test_minimal_nsgaii(self):
        """Minimal NSGAII run should succeed."""
        from vamos import run_nsgaii

        result = run_nsgaii("zdt1", max_evaluations=50, pop_size=10)
        assert len(result) > 0

    def test_minimal_run_dispatch(self):
        """run() should work with minimal args."""
        from vamos import run

        result = run("zdt1", "nsgaii", max_evaluations=50, pop_size=10)
        assert len(result) > 0


class TestQuickResultMethods:
    """Test QuickResult helper methods."""

    def test_summary_runs(self):
        """summary() should print without error."""
        from vamos import run_nsgaii

        result = run_nsgaii("zdt1", max_evaluations=100, pop_size=10)
        # Should not raise
        result.summary()

    @pytest.mark.skipif(
        not _has_matplotlib(),
        reason="matplotlib not installed",
    )
    def test_plot_runs(self):
        """plot() should work when matplotlib is available."""
        from vamos import run_nsgaii

        result = run_nsgaii("zdt1", max_evaluations=100, pop_size=10)
        ax = result.plot(show=False)
        assert ax is not None

    @pytest.mark.skipif(
        not _has_pandas(),
        reason="pandas not installed",
    )
    def test_to_dataframe_runs(self):
        """to_dataframe() should work when pandas is available."""
        from vamos import run_nsgaii

        result = run_nsgaii("zdt1", max_evaluations=100, pop_size=10)
        df = result.to_dataframe()
        assert len(df) == len(result)
        # Uses 1-based naming: f1, f2, ...
        assert "f1" in df.columns
        assert "f2" in df.columns
