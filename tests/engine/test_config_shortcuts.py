"""Tests for config shortcut methods (default)."""

from __future__ import annotations

from math import comb

from vamos.engine.api import MOEADConfig, NSGAIIConfig, NSGAIIIConfig, SMSEMOAConfig, SPEA2Config


class TestNSGAIIConfigShortcuts:
    """Test NSGAIIConfig.default()."""

    def test_default_creates_valid_config(self):
        """default() should create a valid frozen config."""
        cfg = NSGAIIConfig.default()

        assert cfg.pop_size == 100
        assert cfg.crossover[0] == "sbx"
        assert cfg.mutation[0] == "pm"
        assert cfg.selection[0] == "tournament"

    def test_default_with_custom_pop_size(self):
        """default() should accept custom pop_size."""
        cfg = NSGAIIConfig.default(pop_size=50)
        assert cfg.pop_size == 50

    def test_default_with_n_var(self):
        """default() should set mutation prob based on n_var."""
        cfg = NSGAIIConfig.default(n_var=30)
        # mutation prob should be ~1/30 = 0.033
        assert abs(cfg.mutation[1]["prob"] - 1 / 30) < 0.001


class TestMOEADConfigShortcuts:
    """Test MOEADConfig.default()."""

    def test_default_creates_valid_config(self):
        """default() should create a valid frozen config."""
        cfg = MOEADConfig.default()

        assert cfg.pop_size == 91
        assert cfg.neighbor_size == 20
        assert cfg.delta == 0.9
        assert cfg.replace_limit == 2
        assert cfg.aggregation[0] == "pbi"


class TestSPEA2ConfigShortcuts:
    """Test SPEA2Config.default()."""

    def test_default_creates_valid_config(self):
        """default() should create a valid frozen config."""
        cfg = SPEA2Config.default()

        assert cfg.pop_size == 100
        assert cfg.archive_size == 100
        assert cfg.crossover[0] == "sbx"
        assert cfg.selection[0] == "tournament"


class TestSMSEMOAConfigShortcuts:
    """Test SMSEMOAConfig.default()."""

    def test_default_creates_valid_config(self):
        """default() should create a valid frozen config."""
        cfg = SMSEMOAConfig.default()

        assert cfg.pop_size == 100
        assert cfg.crossover[0] == "sbx"
        assert cfg.reference_point["adaptive"] is True


class TestNSGAIIIConfigShortcuts:
    """Test NSGAIIIConfig.default()."""

    def test_default_creates_valid_config(self):
        """default() should create a valid frozen config."""
        cfg = NSGAIIIConfig.default()

        assert cfg.pop_size == comb(12 + 3 - 1, 3 - 1)
        assert cfg.crossover[0] == "sbx"
        assert cfg.reference_directions["divisions"] == 12
