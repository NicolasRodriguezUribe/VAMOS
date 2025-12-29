"""Tests for run_optimization() and unified backend parameter."""

from __future__ import annotations

import numpy as np
import pytest


class TestRunOptimization:
    """Test the simplified run_optimization() function."""

    @pytest.mark.smoke
    def test_run_nsgaii_basic(self):
        """run_optimization() should work with NSGAII."""
        from vamos import run_optimization, ZDT1

        problem = ZDT1(n_var=10)
        result = run_optimization(problem, "nsgaii", max_evaluations=500, pop_size=20)

        assert result.F is not None
        assert result.F.shape[0] > 0
        assert result.F.shape[1] == 2

    @pytest.mark.smoke
    def test_run_moead_basic(self):
        """run_optimization() should work with MOEAD."""
        from vamos import run_optimization, ZDT1

        problem = ZDT1(n_var=10)
        result = run_optimization(problem, "moead", max_evaluations=500, pop_size=50)

        assert result.F is not None
        assert result.F.shape[0] > 0

    @pytest.mark.smoke
    def test_run_spea2_basic(self):
        """run_optimization() should work with SPEA2."""
        from vamos import run_optimization, ZDT1

        problem = ZDT1(n_var=10)
        result = run_optimization(problem, "spea2", max_evaluations=500, pop_size=20)

        assert result.F is not None
        assert result.F.shape[0] > 0

    @pytest.mark.smoke
    def test_run_smsemoa_basic(self):
        """run_optimization() should work with SMSEMOA."""
        from vamos import run_optimization, ZDT1

        problem = ZDT1(n_var=10)
        result = run_optimization(problem, "smsemoa", max_evaluations=500, pop_size=20)

        assert result.F is not None
        assert result.F.shape[0] > 0

    def test_run_invalid_algorithm(self):
        """run_optimization() with invalid algorithm should raise."""
        from vamos import run_optimization, ZDT1
        from vamos.exceptions import InvalidAlgorithmError

        problem = ZDT1(n_var=10)
        with pytest.raises(InvalidAlgorithmError, match="Unknown algorithm"):
            run_optimization(problem, "invalid_algo", max_evaluations=100)

    @pytest.mark.smoke
    def test_result_has_helper_methods(self):
        """Result should have all helper methods."""
        from vamos import run_optimization, ZDT1

        problem = ZDT1(n_var=10)
        result = run_optimization(problem, "nsgaii", max_evaluations=500, pop_size=20)

        # Check helper methods exist
        assert hasattr(result, "summary")
        assert hasattr(result, "best")
        assert hasattr(result, "plot")
        assert hasattr(result, "to_dataframe")
        assert hasattr(result, "save")

    @pytest.mark.smoke
    def test_run_with_different_seeds(self):
        """Different seeds should produce different results."""
        from vamos import run_optimization, ZDT1

        problem = ZDT1(n_var=10)
        result1 = run_optimization(
            problem, "nsgaii", max_evaluations=500, pop_size=20, seed=42
        )
        result2 = run_optimization(
            problem, "nsgaii", max_evaluations=500, pop_size=20, seed=123
        )

        # Results should differ (not exactly equal)
        assert result1.F.shape != result2.F.shape or not np.allclose(result1.F, result2.F)


class TestUnifiedBackendParameter:
    """Test the unified engine parameter."""

    @pytest.mark.smoke
    def test_optimize_engine_override(self):
        """optimize() should accept engine parameter override."""
        from vamos import optimize, OptimizeConfig, NSGAIIConfig, ZDT1

        problem = ZDT1(n_var=10)
        cfg = OptimizeConfig(
            problem=problem,
            algorithm="nsgaii",
            algorithm_config=NSGAIIConfig.default(pop_size=20, n_var=10),
            termination=("n_eval", 500),
            seed=42,
        )

        # Pass engine as override
        result = optimize(cfg, engine="numpy")

        assert result.F is not None
        assert result.F.shape[0] > 0

    @pytest.mark.smoke
    def test_run_optimization_engine_parameter(self):
        """run_optimization() should accept engine parameter."""
        from vamos import run_optimization, ZDT1

        problem = ZDT1(n_var=10)
        result = run_optimization(
            problem, "nsgaii", max_evaluations=500, pop_size=20, engine="numpy"
        )

        assert result.F is not None

    @pytest.mark.smoke
    def test_config_default_engine(self):
        """Config.default() should accept engine parameter."""
        from vamos import NSGAIIConfig

        cfg = NSGAIIConfig.default(pop_size=50, n_var=10, engine="numpy")

        # Verify the config has the engine set
        assert cfg.engine == "numpy"

    @pytest.mark.smoke
    def test_moead_config_default_engine(self):
        """MOEADConfig.default() should accept engine parameter."""
        from vamos import MOEADConfig

        cfg = MOEADConfig.default(pop_size=50, n_var=10, engine="numpy")

        assert cfg.engine == "numpy"


class TestAPIConsistency:
    """Test API consistency across interfaces."""

    @pytest.mark.smoke
    def test_run_optimization_returns_optimization_result(self):
        """run_optimization() should return OptimizationResult."""
        from vamos import run_optimization, OptimizationResult, ZDT1

        problem = ZDT1(n_var=10)
        result = run_optimization(problem, "nsgaii", max_evaluations=500, pop_size=20)

        assert isinstance(result, OptimizationResult)

    @pytest.mark.smoke
    def test_all_exports_available(self):
        """All new exports should be available from vamos."""
        from vamos import (
            run_optimization,
            optimize,
            NSGAIIConfig,
            MOEADConfig,
        )

        # Just verify imports work
        assert run_optimization is not None
        assert optimize is not None
        assert NSGAIIConfig.default is not None
        assert MOEADConfig.default is not None
