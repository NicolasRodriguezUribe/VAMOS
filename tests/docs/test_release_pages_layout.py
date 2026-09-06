from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_versioned_pages_canonical_urls_resolve_to_deployed_files(tmp_path: Path) -> None:
    pytest.importorskip("mkdocs")
    output = tmp_path / "public"
    result = subprocess.run(
        [sys.executable, "tools/build_release_docs.py", "--version", "1.0.0", "--output", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    count = 0
    for page in output.rglob("*.html"):
        for canonical in re.findall(r'<link\s+rel="canonical"\s+href="([^"]+)"', page.read_text(encoding="utf-8")):
            url = urlsplit(canonical)
            assert url.scheme == "https" and url.netloc == "vamos-optimization.github.io"
            assert url.path.startswith("/VAMOS/1.0.0/") or url.path.startswith("/VAMOS/website/")
            target = output / unquote(url.path.removeprefix("/VAMOS/"))
            if url.path.endswith("/"):
                target /= "index.html"
            assert target.is_file(), canonical
            count += 1
    assert count > 80
    assert "url=latest/" in (output / "index.html").read_text(encoding="utf-8")
    assert (output / "latest/index.html").read_bytes() == (output / "1.0.0/index.html").read_bytes()
    homepage = (output / "1.0.0/index.html").read_text(encoding="utf-8")
    assert "https://github.com/vamos-optimization/VAMOS/blob/main/CITATION.cff" in homepage
    assert "https://github.com/vamos-optimization/VAMOS/blob/main/SECURITY.md" in homepage
