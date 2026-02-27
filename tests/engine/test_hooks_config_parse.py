import pytest

from vamos.engine.hooks.config_parse import build_archive_cfg


def test_build_archive_cfg_accepts_simplified_fields():
    cfg = build_archive_cfg({"size_cap": 120, "prune_policy": "spea2"})
    assert cfg.size_cap == 120
    assert cfg.prune_policy == "spea2"


def test_build_archive_cfg_rejects_legacy_fields():
    with pytest.raises(TypeError):
        build_archive_cfg({"archive_type": "size_cap"})
