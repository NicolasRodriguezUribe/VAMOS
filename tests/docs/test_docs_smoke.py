from __future__ import annotations

from pathlib import Path

import pytest

from .smoke_manifest import DOC_SMOKE_CASES, DocSmokeCase


@pytest.mark.smoke
@pytest.mark.parametrize("case", DOC_SMOKE_CASES, ids=lambda case: case.name)
def test_doc_smoke_cases(case: DocSmokeCase) -> None:
    assert Path(case.source_path).exists(), f"Missing doc source for smoke case: {case.source_path}"
    namespace = {"__name__": "__main__"}
    exec(compile(case.code, case.source_path, "exec"), namespace)  # noqa: S102
