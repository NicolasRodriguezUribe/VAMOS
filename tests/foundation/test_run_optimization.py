"""Tests for optimize() convenience path and unified backend parameter."""

from __future__ import annotations

import numpy as np
import pytest

from vamos.api import OptimizeConfig, OptimizationResult, optimize
from vamos.engine.api import MOEADConfig, NSGAIIConfig
from vamos.foundation.exceptions import InvalidAlgorithmError
from vamos.foundation.problems_registry import ZDT1


class TestOptimizeConvenience:
    """Test the unified optimize() convenience path."""

    @pytest.mark.smoke
    def test_run_nsgaii_basic(self):
        """optimize() should work with NSGAII."""
        problem = ZDT1(n_var=10)
        result = optimize(problem, algorithm="nsgaii", budget=500, pop_size=20)

        assert result.F is not None
        assert result.F.shape[0] > 0
        assert result.F.shape[1] == 2

    @pytest.mark.smoke
    def test_run_moead_basic(self):
        """optimize() should work with MOEAD."""
        problem = ZDT1(n_var=10)
        result = optimize(problem, algorithm="moead", budget=500, pop_size=50)

        assert result.F is not None
        assert result.F.shape[0] > 0

    @pytest.mark.smoke
    def test_run_spea2_basic(self):
        """optimize() should work with SPEA2."""
        problem = ZDT1(n_var=10)
        result = optimize(problem, algorithm="spea2", budget=500, pop_size=20)

        assert result.F is not None
        assert result.F.shape[0] > 0

    @pytest.mark.smoke
    def test_run_smsemoa_basic(self):
        """optimize() should work with SMSEMOA."""
        problem = ZDT1(n_var=10)
        result = optimize(problem, algorithm="smsemoa", budget=500, pop_size=20)

        assert result.F is not None
        assert result.F.shape[0] > 0

    def test_run_invalid_algorithm(self):
        """optimize() with invalid algorithm should raise."""
        problem = ZDT1(n_var=10)
        with pytest.raises(InvalidAlgorithmError, match="Unknown algorithm"):
            optimize(problem, algorithm="invalid_algo", budget=100, pop_size=20)

    @pytest.mark.smoke
    def test_result_has_helper_methods(self):
        """Result should have all helper methods."""
        problem = ZDT1(n_var=10)
        result = optimize(problem, algorithm="nsgaii", budget=500, pop_size=20)

        # Check helper methods exist
        assert hasattr(result, "summary")
        assert hasattr(result, "best")
        assert hasattr(result, "plot")
        assert hasattr(result, "to_dataframe")
        assert hasattr(result, "save")

    @pytest.mark.smoke
    def test_run_with_different_seeds(self):
        """Different seeds should produce different results."""
        problem = ZDT1(n_var=10)
        result1 = optimize(problem, algorithm="nsgaii", budget=500, pop_size=20, seed=42)
        result2 = optimize(problem, algorithm="nsgaii", budget=500, pop_size=20, seed=123)

        # Results should differ (not exactly equal)
        assert result1.F.shape != result2.F.shape or not np.allclose(result1.F, result2.F)


class TestUnifiedBackendParameter:
    """Test the unified engine parameter."""

    @pytest.mark.smoke
    def test_optimize_engine_override(self):
        """optimize() should accept engine parameter override."""
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
    def test_optimize_engine_parameter(self):
        """optimize() should accept engine parameter."""
        problem = ZDT1(n_var=10)
        result = optimize(problem, algorithm="nsgaii", budget=500, pop_size=20, engine="numpy")

        assert result.F is not None

    @pytest.mark.smoke
    def test_config_default_engine(self):
        """Config.default() should accept engine parameter."""
        cfg = NSGAIIConfig.default(pop_size=50, n_var=10, engine="numpy")

        # Verify the config has the engine set
        assert cfg.engine == "numpy"

    @pytest.mark.smoke
    def test_moead_config_default_engine(self):
        """MOEADConfig.default() should accept engine parameter."""
        cfg = MOEADConfig.default(pop_size=50, n_var=10, engine="numpy")

        assert cfg.engine == "numpy"


class TestAPIConsistency:
    """Test API consistency across interfaces."""

    @pytest.mark.smoke
    def test_optimize_returns_optimization_result(self):
        """optimize() should return OptimizationResult."""
        problem = ZDT1(n_var=10)
        result = optimize(problem, algorithm="nsgaii", budget=500, pop_size=20)

        assert isinstance(result, OptimizationResult)

    @pytest.mark.smoke
    def test_all_exports_available(self):
        """All new exports should be available from canonical facades."""
        assert optimize is not None
        assert NSGAIIConfig.default is not None
        assert MOEADConfig.default is not None
