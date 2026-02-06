from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from vamos.assist.plan import create_plan
from vamos.assist.providers.protocol import ProviderResponse


class _MixedOverrideProvider:
    name = "mixed_override_test"

    def propose(
        self,
        prompt: str,
        catalog: Mapping[str, object],
        templates: list[str],
        problem_type_hint: str | None = None,
        answers: dict[str, str] | None = None,
    ) -> ProviderResponse:
        del prompt
        del catalog
        del problem_type_hint
        del answers
        template = "demo" if "demo" in templates else templates[0]
        return {
            "kind": "plan",
            "template": template,
            "problem_type": "real",
            "overrides": {
                "max_evaluations": 321,
                "definitely_not_allowed": 999,
            },
            "warnings": [],
        }


def test_auto_mode_filters_disallowed_overrides(tmp_path: Path) -> None:
    plan_dir = create_plan(
        prompt="Apply mixed overrides safely.",
        template=None,
        problem_type="real",
        out_dir=tmp_path / "assist_plan_auto_override_filter",
        mode="auto",
        provider=_MixedOverrideProvider(),
        provider_name="mixed_override_test",
    )

    metadata = json.loads((plan_dir / "plan.json").read_text(encoding="utf-8"))
    config = json.loads((plan_dir / "config.json").read_text(encoding="utf-8"))

    auto = metadata.get("auto")
    assert isinstance(auto, dict)
    assert auto.get("overrides_requested") == {
        "max_evaluations": 321,
        "definitely_not_allowed": 999,
    }
    assert auto.get("overrides_applied") == {"max_evaluations": 321}
    assert auto.get("overrides_rejected") == ["definitely_not_allowed"]
    warnings = metadata.get("warnings")
    assert isinstance(warnings, list)
    assert any("definitely_not_allowed" in str(item) for item in warnings)

    defaults = config.get("defaults")
    assert isinstance(defaults, dict)
    assert defaults.get("max_evaluations") == 321
    assert "definitely_not_allowed" not in defaults
