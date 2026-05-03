from __future__ import annotations

from pathlib import Path

BUDGET_PATH = Path(__file__).with_name("mypy_error_budget.json")
CI_PATH = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"


def test_mypy_error_budget_history_exists() -> None:
    assert BUDGET_PATH.exists()


def test_ci_mypy_scope_covers_strict_public_layers() -> None:
    workflow = CI_PATH.read_text(encoding="utf-8")

    assert "mypy --config-file pyproject.toml" in workflow
    for path in (
        "src/vamos/engine/algorithm/config",
        "src/vamos/engine/algorithm/registry.py",
        "src/vamos/engine/config/spec.py",
        "src/vamos/foundation/eval",
        "src/vamos/experiment/optimization_result",
        "src/vamos/experiment/unified.py",
    ):
        assert path in workflow
