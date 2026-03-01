from __future__ import annotations

from vamos.foundation.core import experiment_config as cfg


def test_resolve_engine_preserves_explicit_vamos_numba() -> None:
    assert cfg.resolve_engine("vamos-numba", algorithm="nsgaii") == "vamos-numba"


def test_resolve_engine_auto_does_not_choose_numba_mixed(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "_has_cpp", lambda: False)
    monkeypatch.setattr(cfg, "_has_numba", lambda: True)
    assert cfg.resolve_engine(None, algorithm="nsgaii") == "numba"
