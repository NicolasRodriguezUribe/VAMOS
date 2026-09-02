from __future__ import annotations

from pathlib import Path


def test_studio_uses_canonical_study_traversal_not_run_manifest_discovery() -> None:
    root = Path(__file__).resolve().parents[2]
    data_source = (root / "src/vamos/ux/studio/data.py").read_text(encoding="utf-8")
    services_source = (root / "src/vamos/ux/studio/services.py").read_text(encoding="utf-8")
    assert 'rglob("manifest.json")' not in data_source
    assert 'rglob("manifest.json")' not in services_source
    assert "load_study(" in data_source
    assert "study.summarize()" in data_source
