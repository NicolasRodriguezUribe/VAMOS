from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_website_uses_installed_i18n_plugin_directly() -> None:
    requirements = (ROOT / "website" / "requirements.txt").read_text(encoding="utf-8")
    config = (ROOT / "website" / "mkdocs.yml").read_text(encoding="utf-8")

    assert "mkdocs-static-i18n>=1.2" in requirements
    assert "- i18n:" in config
    assert "docs_structure: folder" in config
    assert "- locale: en" in config
    assert "default: true" in config
    compatibility_path = ROOT / "website" / "_compat"
    assert not compatibility_path.exists() or not any(item.is_file() for item in compatibility_path.rglob("*"))
