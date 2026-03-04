"""Theme, CSS and navigation helpers for the VAMOS Studio Panel app."""

from __future__ import annotations

import panel as pn

# ---------------------------------------------------------------------------
# Visual system — mirrored from the original Streamlit theme
# ---------------------------------------------------------------------------
ACCENT = "#0f6cbd"
ACCENT_WARM = "#c86d1f"
BG_CARD = "rgba(255,255,255,0.62)"
BORDER = "rgba(15,23,42,0.12)"
TEXT_SOFT = "#445164"
FONT_STACK = '"Aptos", "Segoe UI", system-ui, sans-serif'
DISPLAY_FONT = '"Aptos Display", "Bahnschrift", "Aptos", system-ui, sans-serif'

_GLOBAL_CSS = f"""
:root {{
    --vamos-accent: {ACCENT};
    --vamos-accent-warm: {ACCENT_WARM};
    --vamos-bg-card: {BG_CARD};
    --vamos-border: {BORDER};
    --vamos-text-soft: {TEXT_SOFT};
}}
body {{
    font-family: {FONT_STACK};
}}
h1, h2, h3 {{
    font-family: {DISPLAY_FONT};
    letter-spacing: -0.02em;
}}
.bk-root .bk-btn-primary {{
    background-color: var(--vamos-accent);
}}
.sidenav-btn {{
    width: 100%;
    text-align: left;
    margin-bottom: 4px;
}}
"""

# ---------------------------------------------------------------------------
# Page registry — each entry maps a slug to (label, icon)
# ---------------------------------------------------------------------------
PAGES: dict[str, tuple[str, str]] = {
    "welcome": ("Welcome", ""),
    "problem_builder": ("Problem Builder", ""),
    "experiment": ("Run Experiment", ""),
    "batch": ("Batch Comparison", ""),
    "results": ("Explore Results", ""),
}


def build_template(title: str = "VAMOS Studio") -> pn.template.FastListTemplate:
    """Return a configured FastListTemplate with the VAMOS theme."""
    return pn.template.FastListTemplate(
        title=title,
        site="VAMOS",
        accent_base_color=ACCENT,
        header_background=ACCENT,
        sidebar_width=310,
        main_layout="card",
        raw_css=[_GLOBAL_CSS],
    )


def build_sidebar_nav(current: str = "welcome") -> pn.Column:
    """Build the sidebar navigation column. Returns a Column of buttons."""
    buttons: list[pn.widgets.Button] = []
    for slug, (label, icon) in PAGES.items():
        display = f"{icon}  {label}" if icon else label
        btn = pn.widgets.Button(
            name=display,
            button_type="primary" if slug == current else "light",
            css_classes=["sidenav-btn"],
            sizing_mode="stretch_width",
        )
        btn.tags = [slug]  # used to identify page on click
        buttons.append(btn)
    return pn.Column(*buttons, sizing_mode="stretch_width")
