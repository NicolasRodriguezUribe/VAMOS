import pytest

from vamos.engine.hooks.config_parse import build_archive_cfg


def test_build_archive_cfg_accepts_simplified_fields():
    cfg = build_archive_cfg({"capacity": 120, "pruning": "knn"})
    assert cfg.capacity == 120
    assert cfg.pruning == "knn"


def test_build_archive_cfg_rejects_unknown_fields():
    with pytest.raises(TypeError, match="Unknown archive.external fields"):
        build_archive_cfg({"unexpected_option": True})
