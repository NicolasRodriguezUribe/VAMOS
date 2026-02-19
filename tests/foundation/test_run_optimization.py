"""Tests for optimize() convenience path and unified backend parameter."""

from __future__ import annotations

import numpy as np
import pytest

from vamos.algorithms import MOEADConfig, NSGAIIConfig
from vamos.api import OptimizationResult, optimize
from vamos.foundation.exceptions import InvalidAlgorithmError
from vamos.foundation.problem.zdt1 import ZDT1Problem as ZDT1


class TestOptimizeConvenience:
    """Test the unified optimize() convenience path."""

    @pytest.mark.smoke
    def test_run_nsgaii_basic(self):
        """optimize() should work with NSGAII."""
        problem = ZDT1(n_var=10)
        result = optimize(problem, algorithm="nsgaii", max_evaluations=500, pop_size=20)

        assert result.F is not None
        assert result.F.shape[0] > 0
        assert result.F.shape[1] == 2

    @pytest.mark.smoke
    def test_run_string_problem_basic(self):
        """optimize() should work with a registry problem name + params."""
        result = optimize("zdt1", algorithm="nsgaii", max_evaluations=200, pop_size=20, seed=42, n_var=10)

        assert result.F is not None
        assert result.X is not None
        assert result.X.shape[1] == 10

    @pytest.mark.smoke
    def test_run_moead_basic(self):
        """optimize() should work with MOEAD."""
        problem = ZDT1(n_var=10)
        result = optimize(problem, algorithm="moead", max_evaluations=500, pop_size=50)

        assert result.F is not None
        assert result.F.shape[0] > 0

    @pytest.mark.smoke
    def test_run_spea2_basic(self):
        """optimize() should work with SPEA2."""
        problem = ZDT1(n_var=10)
        result = optimize(problem, algorithm="spea2", max_evaluations=500, pop_size=20)

        assert result.F is not None
        assert result.F.shape[0] > 0

    @pytest.mark.smoke
    def test_run_smsemoa_basic(self):
        """optimize() should work with SMSEMOA."""
        problem = ZDT1(n_var=10)
        result = optimize(problem, algorithm="smsemoa", max_evaluations=500, pop_size=20)

        assert result.F is not None
        assert result.F.shape[0] > 0

    def test_run_invalid_algorithm(self):
        """optimize() with invalid algorithm should raise."""
        problem = ZDT1(n_var=10)
        with pytest.raises(InvalidAlgorithmError, match="Unknown algorithm"):
            optimize(problem, algorithm="invalid_algo", max_evaluations=100, pop_size=20)

    def test_rejects_unknown_algorithm_kwargs(self):
        """optimize() should not silently ignore unknown algorithm kwargs."""
        problem = ZDT1(n_var=10)
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            optimize(problem, algorithm="nsgaii", max_evaluations=100, pop_size=20, unknown_param=1)

    @pytest.mark.smoke
    def test_result_has_helper_methods(self):
        """Result should expose selection helpers and UX helpers should be importable."""
        from vamos.ux.api import (
            plot_result_front,
            result_summary_text,
            result_to_dataframe,
            save_result,
        )

        problem = ZDT1(n_var=10)
        result = optimize(problem, algorithm="nsgaii", max_evaluations=500, pop_size=20)

        assert hasattr(result, "best")
        assert hasattr(result, "front")
        assert callable(result_summary_text)
        assert callable(plot_result_front)
        assert callable(result_to_dataframe)
        assert callable(save_result)

    @pytest.mark.smoke
    def test_run_with_different_seeds(self):
        """Different seeds should produce different results."""
        problem = ZDT1(n_var=10)
        result1 = optimize(problem, algorithm="nsgaii", max_evaluations=500, pop_size=20, seed=42)
        result2 = optimize(problem, algorithm="nsgaii", max_evaluations=500, pop_size=20, seed=123)

        # Results should differ (not exactly equal)
        assert result1.F.shape != result2.F.shape or not np.allclose(result1.F, result2.F)


class TestUnifiedBackendParameter:
    """Test the unified engine parameter."""

    @pytest.mark.smoke
    def test_optimize_engine_override(self):
        """optimize() should accept engine parameter with explicit configs."""
        problem = ZDT1(n_var=10)
        cfg = NSGAIIConfig.default(pop_size=20, n_var=10)
        result = optimize(
            problem,
            algorithm="nsgaii",
            algorithm_config=cfg,
            termination=("max_evaluations", 500),
            seed=42,
            engine="numpy",
        )

        assert result.F is not None
        assert result.F.shape[0] > 0

    @pytest.mark.smoke
    def test_optimize_engine_parameter(self):
        """optimize() should accept engine parameter."""
        problem = ZDT1(n_var=10)
        result = optimize(problem, algorithm="nsgaii", max_evaluations=500, pop_size=20, engine="numpy")

        assert result.F is not None

    @pytest.mark.smoke
    def test_config_default_rejects_engine_parameter(self):
        """Algorithm config defaults should not accept engine (engine is run-level)."""
        with pytest.raises(TypeError):
            _ = NSGAIIConfig.default(pop_size=50, n_var=10, engine="numpy")  # type: ignore[call-arg]

    @pytest.mark.smoke
    def test_moead_config_default_rejects_engine_parameter(self):
        """Algorithm config defaults should not accept engine (engine is run-level)."""
        with pytest.raises(TypeError):
            _ = MOEADConfig.default(pop_size=50, n_var=10, engine="numpy")  # type: ignore[call-arg]


class TestAPIConsistency:
    """Test API consistency across interfaces."""

    @pytest.mark.smoke
    def test_optimize_returns_optimization_result(self):
        """optimize() should return OptimizationResult."""
        problem = ZDT1(n_var=10)
        result = optimize(problem, algorithm="nsgaii", max_evaluations=500, pop_size=20)

        assert isinstance(result, OptimizationResult)

    @pytest.mark.smoke
    def test_all_exports_available(self):
        """All new exports should be available from canonical facades."""
        assert optimize is not None
        assert NSGAIIConfig.default is not None
        assert MOEADConfig.default is not None
