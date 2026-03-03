"""Shared shell and theme helpers for the Studio app."""

from __future__ import annotations

from typing import Any

_CUSTOM_CSS = """\
<style>
/* --- Visual system --- */
:root {
    --vamos-accent: #0f6cbd;
    --vamos-accent-soft: rgba(15,108,189,0.16);
    --vamos-accent-warm: #c86d1f;
    --vamos-bg-card: rgba(255,255,255,0.62);
    --vamos-bg-card-strong: rgba(255,255,255,0.82);
    --vamos-border: rgba(15,23,42,0.12);
    --vamos-shadow: 0 18px 40px rgba(15,23,42,0.08);
    --vamos-text-soft: #445164;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top left, rgba(15,108,189,0.14), transparent 34%),
        radial-gradient(circle at top right, rgba(200,109,31,0.12), transparent 28%),
        linear-gradient(180deg, rgba(248,250,252,0.96), rgba(242,246,250,0.96));
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stAppViewBlockContainer"] {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(15,108,189,0.10), transparent 34%),
        linear-gradient(180deg, rgba(255,255,255,0.92), rgba(246,248,251,0.92));
    border-right: 1px solid var(--vamos-border);
}

html, body, [class*="css"] {
    font-family: "Aptos", "Segoe UI", sans-serif;
}

h1, h2, h3, .stTabs [role="tab"], .studio-kicker {
    font-family: "Aptos Display", "Bahnschrift", "Aptos", sans-serif;
    letter-spacing: -0.02em;
}

.studio-hero {
    padding: 1.4rem 1.5rem 1.5rem 1.5rem;
    margin: 0.2rem 0 1.2rem 0;
    border: 1px solid var(--vamos-border);
    border-radius: 24px;
    background:
        linear-gradient(135deg, rgba(15,108,189,0.18), rgba(200,109,31,0.14)),
        var(--vamos-bg-card-strong);
    box-shadow: var(--vamos-shadow);
}

.studio-hero h2 {
    margin: 0.2rem 0 0.35rem 0;
    font-size: clamp(1.8rem, 3vw, 2.5rem);
}

.studio-hero p {
    margin: 0;
    max-width: 52rem;
    color: var(--vamos-text-soft);
    font-size: 1rem;
}

.studio-kicker {
    display: inline-block;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--vamos-accent);
}

.studio-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin-top: 1rem;
}

.studio-chip {
    display: inline-flex;
    align-items: center;
    padding: 0.42rem 0.8rem;
    border-radius: 999px;
    border: 1px solid rgba(15,23,42,0.08);
    background: rgba(255,255,255,0.72);
    color: #1f2937;
    font-size: 0.88rem;
}

.studio-section-intro {
    margin: 0.25rem 0 1rem 0;
    padding: 0.95rem 1rem;
    border: 1px solid var(--vamos-border);
    border-radius: 20px;
    background: var(--vamos-bg-card);
    box-shadow: 0 8px 24px rgba(15,23,42,0.04);
}

.studio-section-intro h3 {
    margin: 0.15rem 0 0.3rem 0;
    font-size: 1.2rem;
}

.studio-section-intro p {
    margin: 0;
    color: var(--vamos-text-soft);
}

.studio-step-card {
    padding: 1rem 1rem 0.15rem 1rem;
    margin-bottom: 0.75rem;
    border: 1px solid var(--vamos-border);
    border-radius: 18px;
    background: var(--vamos-bg-card);
    box-shadow: 0 10px 28px rgba(15,23,42,0.04);
}

.studio-step-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2rem;
    height: 2rem;
    border-radius: 999px;
    font-weight: 700;
    color: white;
    background: linear-gradient(135deg, var(--vamos-accent), var(--vamos-accent-warm));
}

.studio-data-strip {
    margin: 0.35rem 0 1rem 0;
    padding: 0.8rem 1rem;
    border-radius: 18px;
    border: 1px solid var(--vamos-border);
    background: rgba(255,255,255,0.68);
}

.studio-data-strip p {
    margin: 0;
    color: var(--vamos-text-soft);
}

div[data-baseweb="tab-list"] {
    gap: 0.45rem;
    padding: 0.35rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.7);
    border: 1px solid rgba(15,23,42,0.08);
    margin-bottom: 1rem;
}

[role="tab"] {
    border-radius: 999px !important;
    border: 1px solid transparent !important;
    padding: 0.45rem 0.9rem !important;
    transition: background-color 0.15s ease, color 0.15s ease, transform 0.15s ease;
}

[role="tab"][aria-selected="true"] {
    color: white !important;
    background: linear-gradient(135deg, var(--vamos-accent), #1381b2) !important;
    box-shadow: 0 10px 18px rgba(15,108,189,0.18);
}

[role="tab"]:hover {
    transform: translateY(-1px);
}

div[data-testid="stExpander"] {
    border: 1px solid rgba(15,23,42,0.10) !important;
    border-radius: 18px !important;
    background: rgba(255,255,255,0.68);
    overflow: hidden;
}

div[data-testid="stExpander"] summary p {
    font-weight: 600;
}

[data-testid="metric-container"] {
    border: 1px solid rgba(15,23,42,0.08);
    border-radius: 18px;
    background: linear-gradient(180deg, rgba(255,255,255,0.82), rgba(250,250,252,0.78));
    box-shadow: 0 10px 24px rgba(15,23,42,0.04);
    padding: 0.7rem 0.8rem;
}

.stButton > button, .stDownloadButton > button {
    border-radius: 999px;
    border: 1px solid rgba(15,23,42,0.08);
    box-shadow: 0 10px 20px rgba(15,23,42,0.05);
}

[data-testid="stAlert"] {
    border-radius: 18px;
}

.kbd-hint {
    display: inline-block;
    padding: 2px 7px;
    font-size: 0.75rem;
    font-family: monospace;
    background: var(--vamos-bg-card);
    border: 1px solid var(--vamos-border);
    border-radius: 4px;
    margin-left: 4px;
    vertical-align: middle;
}

.walkthrough-banner {
    padding: 1.2rem 1.5rem;
    border-left: 4px solid var(--vamos-accent);
    background: var(--vamos-bg-card);
    border-radius: 0 8px 8px 0;
    margin-bottom: 1rem;
}

.walkthrough-banner h4 { margin: 0 0 0.5rem 0; }

button:focus-visible, input:focus-visible, select:focus-visible,
textarea:focus-visible, [role="tab"]:focus-visible {
    outline: 2px solid var(--vamos-accent) !important;
    outline-offset: 2px;
}

.sr-only {
    position: absolute;
    width: 1px; height: 1px;
    padding: 0; margin: -1px;
    overflow: hidden;
    clip: rect(0,0,0,0);
    border: 0;
}
</style>
"""


def inject_accessibility_css(st: Any) -> None:
    """Inject the shared CSS customizations."""
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


def render_studio_shell(st: Any) -> None:
    """Render the shared Studio banner above the tabs."""
    st.markdown(
        '<div class="studio-hero">'
        '<span class="studio-kicker">Interactive Optimization Workspace</span>'
        "<h2>VAMOS Studio</h2>"
        "<p>Build a problem, run a quick preview, and compare trade-offs without dropping into code until you actually need it.</p>"
        '<div class="studio-chip-row">'
        '<span class="studio-chip">Template-first flow</span>'
        '<span class="studio-chip">Safe defaults</span>'
        '<span class="studio-chip">Built-in demo results</span>'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


__all__ = ["inject_accessibility_css", "render_studio_shell"]
