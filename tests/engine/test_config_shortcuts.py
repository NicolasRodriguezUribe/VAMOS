"""Tests for config shortcut methods (default, from_dict)."""

from __future__ import annotations

import pytest


class TestNSGAIIConfigShortcuts:
    """Test NSGAIIConfig.default() and from_dict()."""

    def test_default_creates_valid_config(self):
        """default() should create a valid frozen config."""
        from vamos import NSGAIIConfig

        cfg = NSGAIIConfig.default()

        assert cfg.pop_size == 100
        assert cfg.crossover[0] == "sbx"
        assert cfg.mutation[0] == "pm"
        assert cfg.selection[0] == "tournament"
        assert cfg.survival == "rank_crowding"
        assert cfg.engine == "numpy"

    def test_default_with_custom_pop_size(self):
        """default() should accept custom pop_size."""
        from vamos import NSGAIIConfig

        cfg = NSGAIIConfig.default(pop_size=50)
        assert cfg.pop_size == 50

    def test_default_with_n_var(self):
        """default() should set mutation prob based on n_var."""
        from vamos import NSGAIIConfig

        cfg = NSGAIIConfig.default(n_var=30)
        # mutation prob should be ~1/30 = 0.033
        assert abs(cfg.mutation[1]["prob"] - 1 / 30) < 0.001

    def test_from_dict_basic(self):
        """from_dict() should create config from dictionary."""
        from vamos import NSGAIIConfig

        cfg = NSGAIIConfig.from_dict(
            {
                "pop_size": 50,
                "crossover": ("sbx", {"prob": 0.8, "eta": 15}),
                "mutation": ("pm", {"prob": 0.1, "eta": 25}),
                "selection": "tournament",
                "survival": "rank_crowding",
                "engine": "numpy",
            }
        )

        assert cfg.pop_size == 50
        assert cfg.crossover == ("sbx", {"prob": 0.8, "eta": 15})
        assert cfg.mutation == ("pm", {"prob": 0.1, "eta": 25})

    def test_from_dict_with_dict_operators(self):
        """from_dict() should handle dict-style operator configs."""
        from vamos import NSGAIIConfig

        cfg = NSGAIIConfig.from_dict(
            {
                "pop_size": 100,
                "crossover": {"method": "sbx", "prob": 0.9},
                "mutation": {"method": "pm", "prob": 0.1},
                "selection": {"method": "tournament"},
                "survival": "rank_crowding",
                "engine": "numpy",
            }
        )

        assert cfg.pop_size == 100
        assert cfg.crossover[0] == "sbx"


class TestMOEADConfigShortcuts:
    """Test MOEADConfig.default() and from_dict()."""

    def test_default_creates_valid_config(self):
        """default() should create a valid frozen config."""
        from vamos import MOEADConfig

        cfg = MOEADConfig.default()

        assert cfg.pop_size == 100
        assert cfg.neighbor_size == 20
        assert cfg.delta == 0.9
        assert cfg.replace_limit == 2
        assert cfg.aggregation[0] == "tchebycheff"

    def test_from_dict_basic(self):
        """from_dict() should create config from dictionary."""
        from vamos import MOEADConfig

        cfg = MOEADConfig.from_dict(
            {
                "pop_size": 50,
                "neighbor_size": 10,
                "delta": 0.8,
                "replace_limit": 3,
                "crossover": ("sbx", {"prob": 0.9}),
                "mutation": ("pm", {"prob": 0.1}),
                "aggregation": "pbi",
                "engine": "numpy",
            }
        )

        assert cfg.pop_size == 50
        assert cfg.neighbor_size == 10
        assert cfg.aggregation[0] == "pbi"


class TestSPEA2ConfigShortcuts:
    """Test SPEA2Config.default()."""

    def test_default_creates_valid_config(self):
        """default() should create a valid frozen config."""
        from vamos import SPEA2Config

        cfg = SPEA2Config.default()

        assert cfg.pop_size == 100
        assert cfg.archive_size == 100
        assert cfg.crossover[0] == "sbx"
        assert cfg.selection[0] == "tournament"


class TestSMSEMOAConfigShortcuts:
    """Test SMSEMOAConfig.default()."""

    def test_default_creates_valid_config(self):
        """default() should create a valid frozen config."""
        from vamos import SMSEMOAConfig

        cfg = SMSEMOAConfig.default()

        assert cfg.pop_size == 100
        assert cfg.crossover[0] == "sbx"
        assert cfg.reference_point["adaptive"] is True


class TestNSGAIIIConfigShortcuts:
    """Test NSGAIIIConfig.default()."""

    def test_default_creates_valid_config(self):
        """default() should create a valid frozen config."""
        from vamos import NSGAIIIConfig

        cfg = NSGAIIIConfig.default()

        assert cfg.pop_size == 92
        assert cfg.crossover[0] == "sbx"
        assert cfg.reference_directions["divisions"] == 12
