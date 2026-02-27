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

_BUILDERS = [
    NSGAIIConfig.builder,
    MOEADConfig.builder,
    SMSEMOAConfig.builder,
    NSGAIIIConfig.builder,
    SPEA2Config.builder,
    IBEAConfig.builder,
    SMPSOConfig.builder,
    AGEMOEAConfig.builder,
    RVEAConfig.builder,
]


@pytest.mark.parametrize("builder_factory", _BUILDERS)
def test_external_archive_builder_accepts_capacity_and_pruning_only(builder_factory):
    builder = builder_factory()
    builder.external_archive(capacity=50, pruning="random")
    ext_cfg = builder._cfg["external_archive"]  # type: ignore[attr-defined]
    assert ext_cfg.capacity == 50
    assert ext_cfg.pruning == "random"


@pytest.mark.parametrize("builder_factory", _BUILDERS)
def test_external_archive_builder_rejects_legacy_kwargs(builder_factory):
    builder = builder_factory()
    with pytest.raises(TypeError):
        builder.external_archive(capacity=50, archive_type="size_cap")  # type: ignore[call-arg]
