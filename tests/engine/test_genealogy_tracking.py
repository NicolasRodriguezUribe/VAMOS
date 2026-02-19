"""Tests for genealogy tracking across all algorithms."""

import pytest

from vamos.engine.algorithm.config import (
    IBEAConfig,
    MOEADConfig,
    NSGAIIConfig,
    NSGAIIIConfig,
    SMPSOConfig,
    SMSEMOAConfig,
    SPEA2Config,
)
from vamos.engine.algorithm.ibea import IBEA
from vamos.engine.algorithm.moead import MOEAD
from vamos.engine.algorithm.nsgaii import NSGAII
from vamos.engine.algorithm.nsgaiii import NSGAIII
from vamos.engine.algorithm.smpso import SMPSO
from vamos.engine.algorithm.smsemoa import SMSEMOA
from vamos.engine.algorithm.spea2 import SPEA2
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.dtlz import DTLZ2Problem
from vamos.foundation.problem.zdt1 import ZDT1Problem


class TestGenealogyTracking:
    """Test genealogy tracking for all algorithms."""

    @pytest.fixture
    def small_problem(self):
        """Small ZDT1 problem for fast tests."""
        return ZDT1Problem(n_var=5)

    @pytest.fixture
    def many_obj_problem(self):
        """Small DTLZ2 problem for NSGA-III tests."""
        return DTLZ2Problem(n_var=6, n_obj=3)

    @pytest.fixture
    def kernel(self):
        """NumPy kernel instance."""
        return NumPyKernel()

    def test_nsgaii_genealogy_tracking(self, small_problem, kernel):
        """Test NSGA-II with genealogy tracking enabled."""
        cfg = (
            NSGAIIConfig.builder()
            .pop_size(10)
            .offspring_size(10)
            .crossover("sbx", prob=0.9, eta=15.0)
            .mutation("pm", prob="1/n", eta=20.0)
            .selection("tournament", pressure=2)
            .track_genealogy(True)
            .build()
        )
        alg = NSGAII(cfg.to_dict(), kernel)
        result = alg.run(small_problem, termination=("max_evaluations", 30), seed=42)

        assert "genealogy" in result
        genealogy = result["genealogy"]
        # Genealogy should contain tracking info (structure may vary)
        assert isinstance(genealogy, dict)
        assert len(genealogy) > 0

    def test_moead_genealogy_tracking(self, small_problem, kernel):
        """Test MOEA/D with genealogy tracking enabled."""
        cfg = (
            MOEADConfig.builder()
            .pop_size(10)
            .crossover("sbx", prob=0.9, eta=15.0)
            .mutation("pm", prob="1/n", eta=20.0)
            .neighbor_size(5)
            .delta(0.9)
            .replace_limit(2)
            .aggregation("tchebycheff")
            .track_genealogy(True)
            .build()
        )
        alg = MOEAD(cfg.to_dict(), kernel)
        result = alg.run(small_problem, termination=("max_evaluations", 30), seed=42)

        assert "genealogy" in result
        genealogy = result["genealogy"]
        assert isinstance(genealogy, dict)
        assert len(genealogy) > 0

    def test_spea2_genealogy_tracking(self, small_problem, kernel):
        """Test SPEA2 with genealogy tracking enabled."""
        cfg = (
            SPEA2Config.builder()
            .pop_size(10)
            .archive_size(10)
            .crossover("sbx", prob=0.9, eta=15.0)
            .mutation("pm", prob="1/n", eta=20.0)
            .selection("tournament", pressure=2)
            .track_genealogy(True)
            .build()
        )
        alg = SPEA2(cfg.to_dict(), kernel)
        result = alg.run(small_problem, termination=("max_evaluations", 30), seed=42)

        assert "genealogy" in result
        genealogy = result["genealogy"]
        assert isinstance(genealogy, dict)
        assert len(genealogy) > 0

    def test_ibea_genealogy_tracking(self, small_problem, kernel):
        """Test IBEA with genealogy tracking enabled."""
        cfg = (
            IBEAConfig.builder()
            .pop_size(10)
            .crossover("sbx", prob=0.9, eta=15.0)
            .mutation("pm", prob="1/n", eta=20.0)
            .selection("tournament", pressure=2)
            .indicator("eps")
            .kappa(0.05)
            .track_genealogy(True)
            .build()
        )
        alg = IBEA(cfg.to_dict(), kernel)
        result = alg.run(small_problem, termination=("max_evaluations", 30), seed=42)

        assert "genealogy" in result
        genealogy = result["genealogy"]
        assert isinstance(genealogy, dict)
        assert len(genealogy) > 0

    def test_smsemoa_genealogy_tracking(self, small_problem, kernel):
        """Test SMS-EMOA with genealogy tracking enabled."""
        cfg = (
            SMSEMOAConfig.builder()
            .pop_size(10)
            .crossover("sbx", prob=0.9, eta=15.0)
            .mutation("pm", prob="1/n", eta=20.0)
            .selection("tournament", pressure=2)
            .track_genealogy(True)
            .build()
        )
        alg = SMSEMOA(cfg.to_dict(), kernel)
        result = alg.run(small_problem, termination=("max_evaluations", 20), seed=42)

        assert "genealogy" in result
        genealogy = result["genealogy"]
        assert isinstance(genealogy, dict)
        assert len(genealogy) > 0

    def test_smpso_genealogy_tracking(self, small_problem, kernel):
        """Test SMPSO with genealogy tracking enabled."""
        cfg = SMPSOConfig.builder().pop_size(10).archive_size(10).mutation("pm", prob="1/n", eta=20.0).track_genealogy(True).build()
        alg = SMPSO(cfg.to_dict(), kernel)
        result = alg.run(small_problem, termination=("max_evaluations", 30), seed=42)

        assert "genealogy" in result
        genealogy = result["genealogy"]
        assert isinstance(genealogy, dict)
        assert len(genealogy) > 0

    def test_nsgaiii_genealogy_tracking(self, many_obj_problem, kernel):
        """Test NSGA-III with genealogy tracking enabled."""
        cfg = (
            NSGAIIIConfig.builder()
            .pop_size(10)
            .crossover("sbx", prob=0.9, eta=15.0)
            .mutation("pm", prob="1/n", eta=20.0)
            .selection("tournament", pressure=2)
            .reference_directions(divisions=3)
            .track_genealogy(True)
            .build()
        )
        alg = NSGAIII(cfg.to_dict(), kernel)
        result = alg.run(many_obj_problem, termination=("max_evaluations", 36), seed=42)

        assert "genealogy" in result
        genealogy = result["genealogy"]
        assert isinstance(genealogy, dict)
        assert len(genealogy) > 0

    def test_genealogy_disabled_by_default(self, small_problem, kernel):
        """Test that genealogy is not tracked when disabled (default)."""
        cfg = (
            NSGAIIConfig.builder()
            .pop_size(10)
            .offspring_size(10)
            .crossover("sbx", prob=0.9, eta=15.0)
            .mutation("pm", prob="1/n", eta=20.0)
            .selection("tournament", pressure=2)
            .build()
        )
        alg = NSGAII(cfg.to_dict(), kernel)
        result = alg.run(small_problem, termination=("max_evaluations", 30), seed=42)

        # Should not have genealogy when disabled
        assert result.get("genealogy") is None or "genealogy" not in result
