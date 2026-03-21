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


@pytest.mark.parametrize("config_cls", ALL_CONFIGS, ids=lambda c: c.__name__)
def test_to_dict_returns_plain_dict(config_cls):
    """to_dict() should return a plain dict, not a dataclass."""
    cfg = config_cls.default()
    d = cfg.to_dict()
    assert isinstance(d, dict)


def test_nsgaii_online_control_roundtrip_preserves_payload() -> None:
    cfg = (
        NSGAIIConfig.builder()
        .pop_size(8)
        .offspring_size(8)
        .crossover("sbx", prob=1.0, eta=20.0)
        .mutation("polynomial", prob=0.1, eta=20.0)
        .selection("tournament", size=2)
        .online_control(enabled=True, trace_level="basic")
        .build()
    )
    cfg2 = NSGAIIConfig.from_dict(cfg.to_dict())
    assert cfg2.online_control == {
        "enabled": True,
        "router": "heuristic",
        "policy": "hierarchical_joint",
        "credit_model": "simple_improvement",
        "trace_level": "basic",
        "prototype_set": "default",
    }


def test_moead_online_control_roundtrip_preserves_payload() -> None:
    cfg = (
        MOEADConfig.builder()
        .pop_size(8)
        .batch_size(1)
        .neighbor_size(3)
        .delta(0.9)
        .replace_limit(1)
        .crossover("de", cr=1.0, f=0.5)
        .mutation("polynomial", prob=0.1, eta=20.0)
        .aggregation("pbi", theta=5.0)
        .online_control(enabled=True, trace_level="basic")
        .build()
    )
    cfg2 = MOEADConfig.from_dict(cfg.to_dict())
    assert cfg2.online_control == {
        "enabled": True,
        "router": "heuristic",
        "policy": "hierarchical_joint",
        "credit_model": "simple_improvement",
        "trace_level": "basic",
        "prototype_set": "default",
    }
