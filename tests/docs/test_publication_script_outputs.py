from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "paper" / "33_update_accessibility_tables.py"
DATASET = ROOT / "paper" / "accessibility_proxy_snippets.json"


def _load_script():
    spec = importlib.util.spec_from_file_location("vamos_paper_accessibility_tables", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_accessibility_metrics_are_deterministic() -> None:
    module = _load_script()
    snippet = "# ignored\nimport numpy as np\n\nvalue = np.ones(2)\n"
    assert module.count_loc(snippet) == 2
    assert module.count_imports(snippet) == 1


def test_accessibility_tables_use_explicit_outputs_and_refuse_collisions(tmp_path: Path) -> None:
    main_out = tmp_path / "generated" / "accessibility_proxies.tex"
    details_out = tmp_path / "generated" / "accessibility_proxy_details.tex"
    command = [
        sys.executable,
        str(SCRIPT),
        "--dataset",
        str(DATASET),
        "--main-out",
        str(main_out),
        "--details-out",
        str(details_out),
    ]

    completed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert main_out.read_text(encoding="utf-8").startswith("\\begin{table*}")
    assert details_out.read_text(encoding="utf-8").startswith("\\begin{table*}")

    main_before = main_out.read_bytes()
    details_before = details_out.read_bytes()
    collision = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert collision.returncode != 0
    assert "Refusing to overwrite" in collision.stderr
    assert main_out.read_bytes() == main_before
    assert details_out.read_bytes() == details_before
