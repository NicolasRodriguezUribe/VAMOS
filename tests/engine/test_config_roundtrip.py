"""Tests for config from_dict/to_dict roundtrip for all 9 algorithms."""

from __future__ import annotations

import pytest

from vamos.engine.algorithm.config import (
    AGEMOEAConfig,
    IBEAConfig,
    MOEADConfig,
    NSGAIIConfig,
    NSGAIIIConfig,
    RVEAConfig,
    SMPSOConfig,
    SMSEMOAConfig,
    SPEA2Config,
)

ALL_CONFIGS = [
    NSGAIIConfig,
    MOEADConfig,
    SPEA2Config,
    SMSEMOAConfig,
    NSGAIIIConfig,
    IBEAConfig,
    SMPSOConfig,
    AGEMOEAConfig,
    RVEAConfig,
]


@pytest.mark.parametrize("config_cls", ALL_CONFIGS, ids=lambda c: c.__name__)
class TestConfigRoundtrip:
    def test_default_roundtrip(self, config_cls):
        """from_dict(cfg.to_dict()) should reconstruct an equal config."""
        cfg = config_cls.default()
        d = cfg.to_dict()
        cfg2 = config_cls.from_dict(d)
        assert cfg == cfg2

    def test_roundtrip_preserves_type(self, config_cls):
        """Reconstructed config should be the same class."""
        cfg = config_cls.default()
        cfg2 = config_cls.from_dict(cfg.to_dict())
        assert type(cfg2) is config_cls

    def test_to_dict_returns_plain_dict(self, config_cls):
        """to_dict() should return a plain dict, not a dataclass."""
        cfg = config_cls.default()
        d = cfg.to_dict()
        assert isinstance(d, dict)
