"""Theme, CSS and navigation helpers for the VAMOS Studio Panel app."""

from __future__ import annotations

import panel as pn

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
ACCENT = "#0d9488"
ACCENT_WARM = "#c86d1f"
FONT_STACK = '"Segoe UI", "Aptos", system-ui, -apple-system, sans-serif'
DISPLAY_FONT = '"Aptos Display", "Bahnschrift", "Segoe UI", system-ui, sans-serif'

_GLOBAL_CSS = f"""
/* ---------- Typography ---------- */
body, #content {{
    font-family: {FONT_STACK};
}}
h1, h2, h3, h4, h5 {{
    font-family: {DISPLAY_FONT};
    letter-spacing: -0.02em;
}}

/* ---------- Buttons ---------- */
.bk-btn-primary {{
    background: {ACCENT} !important;
    border: none !important;
    border-radius: 8px !important;
    color: #fff !important;
    font-weight: 600;
    transition: all 0.2s ease;
}}
.bk-btn-primary:hover {{
    filter: brightness(1.1);
    transform: translateY(-1px);
}}
.bk-btn-success {{
    border-radius: 8px !important;
    font-weight: 600;
}}
.bk-btn-warning {{
    border-radius: 8px !important;
    font-weight: 600;
}}
.bk-btn-light {{
    border-radius: 8px !important;
    transition: all 0.2s ease;
}}

/* ---------- Sidebar nav buttons ---------- */
.sidenav-btn {{
    width: 100%;
    text-align: left;
    margin-bottom: 6px;
    padding: 10px 16px !important;
    font-size: 0.92rem;
}}
.sidenav-btn.bk-btn-primary {{
    border-left: 3px solid {ACCENT} !important;
}}

/* ---------- Inputs ---------- */
.bk-input {{
    border-radius: 6px !important;
}}
.bk-input:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 2px rgba(15,108,189,0.15) !important;
}}

/* ---------- Labels ---------- */
.bk label {{
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.02em;
}}

/* ---------- Alerts ---------- */
.alert-info {{
    border-radius: 8px !important;
}}

/* ---------- Code editor ---------- */
.ace_editor {{
    border-radius: 8px !important;
}}

/* ---------- Tabulator (tables) ---------- */
.pnx-tabulator .tabulator {{
    border-radius: 8px !important;
}}

/* ---------- Plotly ---------- */
.plotly .main-svg {{
    background: transparent !important;
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
    """Return a configured FastListTemplate with the VAMOS theme.

    Light theme by default, with a toggle to switch to dark.
    """
    return pn.template.FastListTemplate(
        title=title,
        site="",
        accent_base_color=ACCENT,
        header_background=ACCENT,
        sidebar_width=310,
        main_layout=None,
        theme="default",
        theme_toggle=True,
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
