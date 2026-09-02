"""Trusted-code and provider controls for the experimental Problem Builder."""

from __future__ import annotations

from typing import Any

import panel as pn

from vamos.ux.studio.problem_builder_backend import TRUSTED_LOCAL_CODE_WARNING


def build_trusted_code_controls(state: Any) -> tuple[Any, Any]:
    warning = pn.pane.Alert(
        f"**Experimental trusted-code feature.** {TRUSTED_LOCAL_CODE_WARNING} "
        "The validation checks reject some unsupported constructs but do not isolate untrusted code.",
        alert_type="danger",
    )
    confirmation = pn.widgets.Checkbox.from_param(
        state.param.trusted_local_code,
        name="I reviewed the complete current code and explicitly trust it for local execution.",
    )
    return warning, confirmation


def build_ai_generation_tab(state: Any) -> Any:
    provider = pn.widgets.Select.from_param(state.param.ai_provider, name="LLM Provider")
    api_key = pn.widgets.PasswordInput.from_param(
        state.param.ai_api_key,
        name="API Key",
        placeholder="Paste your API key here (or set it in the environment)",
    )
    description = pn.widgets.TextAreaInput.from_param(
        state.param.ai_description,
        name="Describe your optimization problem",
        placeholder="Minimize cost and deflection of a beam; include bounds and constraints.",
        height=150,
    )
    generate = pn.widgets.Button(name="Generate Code", button_type="primary")
    generate.on_click(state.ai_generate)
    status = pn.pane.Alert(
        pn.bind(lambda text: text or "Describe your problem and click Generate Code.", state.param.ai_status),
        alert_type="info",
    )
    return pn.Column(
        "### AI-Powered Problem Generation",
        pn.pane.Markdown(
            "Generated objective and constraint code is displayed in **Problem Definition** "
            "and is never executed automatically. Review it completely before opting in."
        ),
        pn.layout.Divider(),
        pn.Row(provider, api_key, sizing_mode="stretch_width"),
        description,
        generate,
        status,
        sizing_mode="stretch_width",
    )


__all__ = ["build_ai_generation_tab", "build_trusted_code_controls"]
