from __future__ import annotations

from pathlib import Path

import pytest

BUDGET_PATH = Path(__file__).with_name("mypy_error_budget.json")

pytestmark = pytest.mark.skip(reason="Historical artifact only. Strict mypy is enforced directly in CI and tools/health.py.")


def test_mypy_error_budget_history_exists() -> None:
    assert BUDGET_PATH.exists()
