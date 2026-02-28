"""VAMOS Studio -- main Streamlit application.

Thin entry point that orchestrates three tabs:

1. **Welcome** -- onboarding wizard (auto-shown on first launch)
2. **Problem Builder** -- interactive problem definition
3. **Explore Results** -- post-run comparison & MCDM
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def _import_streamlit() -> Any:
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("VAMOS Studio requires the 'studio' extra: pip install -e \".[studio]\"") from exc
    return st


def _import_plotly() -> Any:
    try:
        import plotly.express as px
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Plotly is required for interactive plots. Install with the 'studio' extras.") from exc
    return px


# ======================================================================
# Theme / Accessibility CSS
# ======================================================================

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

/* Keyboard shortcut badges */
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

/* Walkthrough banner */
.walkthrough-banner {
    padding: 1.2rem 1.5rem;
    border-left: 4px solid var(--vamos-accent);
    background: var(--vamos-bg-card);
    border-radius: 0 8px 8px 0;
    margin-bottom: 1rem;
}
.walkthrough-banner h4 { margin: 0 0 0.5rem 0; }

/* Better focus outlines for accessibility */
button:focus-visible, input:focus-visible, select:focus-visible,
textarea:focus-visible, [role="tab"]:focus-visible {
    outline: 2px solid var(--vamos-accent) !important;
    outline-offset: 2px;
}

/* Screen-reader-only utility */
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


def _inject_accessibility_css(st: Any) -> None:
    """Inject custom CSS for dark mode awareness, focus outlines, and a11y."""
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


def _render_studio_shell(st: Any) -> None:
    """Render the shared Studio banner above the tab set."""
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


# ======================================================================
# First-launch walkthrough
# ======================================================================


def _render_first_launch_walkthrough(st: Any) -> None:
    """Show a guided walkthrough on the user's first visit.

    Uses ``st.session_state`` to track dismissal so it only appears once
    per session.  The user can dismiss it permanently.
    """
    if st.session_state.get("walkthrough_dismissed", False):
        return

    st.markdown(
        '<div class="walkthrough-banner">'
        "<h4>Welcome! Here's how to get started</h4>"
        "<ol>"
        "<li><b>Problem Builder</b> tab -- choose a starter template and click <b>Run preview</b>. Studio uses safe defaults for you.</li>"
        "<li>Read the chart and summary to see the trade-off between your goals.</li>"
        "<li>Need more control? Turn on <b>Advanced mode</b> later to edit code, change algorithms, or tweak settings.</li>"
        "</ol>"
        "</div>",
        unsafe_allow_html=True,
    )

    if st.button("Got it -- dismiss this guide", key="dismiss_walkthrough"):
        st.session_state["walkthrough_dismissed"] = True
        st.rerun()


# ======================================================================
# Tab: Welcome / Onboarding
# ======================================================================


def _render_welcome_tab(st: Any) -> None:
    """Render the Welcome / Getting Started onboarding page."""
    _render_first_launch_walkthrough(st)

    st.header("Welcome to VAMOS Studio")
    st.markdown(
        '<div class="studio-section-intro">'
        '<span class="studio-kicker">First session</span>'
        "<h3>Start simple, then peel back the layers</h3>"
        "<p>The Studio now defaults to a guided path: start with a template, inspect a preview, then compare solutions only after you have something concrete to look at.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "VAMOS Studio helps you explore trade-offs when you have more than one goal. "
        "You can start with templates and safe defaults, then move into code only when you need it."
    )
    st.caption("Beginner flow: pick a template, run a preview, then inspect the trade-off chart.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            '<div class="studio-step-card"><span class="studio-step-number">1</span></div>',
            unsafe_allow_html=True,
        )
        st.subheader("1. Pick a starter")
        st.markdown(
            "Open **Problem Builder**, choose a template, and stay in the default beginner mode. "
            "You do not need to write Python to get your first Pareto front."
        )
        with st.expander("What beginners change first"):
            st.markdown(
                "- Pick a domain template.\n"
                "- Read the plain-language description.\n"
                "- Click **Run preview**.\n"
                "- Only turn on **Advanced mode** if the template is not enough."
            )

    with col2:
        st.markdown(
            '<div class="studio-step-card"><span class="studio-step-number">2</span></div>',
            unsafe_allow_html=True,
        )
        st.subheader("2. Run a quick preview")
        st.markdown(
            "Studio runs the optimizer with recommended defaults, then shows a chart and summary table. "
            "This is the fastest way to understand the shape of your trade-offs."
        )
        with st.expander("What the preview means"):
            st.markdown(
                "- Each point is one candidate solution.\n"
                "- Better in one goal often means worse in another.\n"
                "- The summary table helps you see the spread of results quickly."
            )

    with col3:
        st.markdown(
            '<div class="studio-step-card"><span class="studio-step-number">3</span></div>',
            unsafe_allow_html=True,
        )
        st.subheader("3. Compare saved runs")
        st.markdown(
            "Use **Explore Results** to compare algorithms, rank solutions based on what matters most to you, "
            "or browse the built-in demo before you have your own results."
        )
        with st.expander("When to use Advanced mode"):
            st.markdown(
                "Turn it on when you want to:\n\n"
                "- edit the objective function\n"
                "- add constraints\n"
                "- change the algorithm\n"
                "- tune budget, seed, or population size"
            )

    st.divider()
    _render_welcome_expanders(st)


def _render_welcome_expanders(st: Any) -> None:
    """Render the collapsible quick-reference sections."""
    # Show the MOO explanation first and auto-expand on first visit
    first_visit = not st.session_state.get("walkthrough_dismissed", False)
    with st.expander("What is multi-objective optimization?", expanded=first_visit):
        st.markdown(
            "**In short:** when you have conflicting goals (e.g. cost vs quality, "
            "speed vs accuracy), there's no single best answer. Instead, you get "
            "a set of **trade-off solutions** called the **Pareto front** -- "
            "improving one goal always means sacrificing another.\n\n"
            "VAMOS uses **evolutionary algorithms** to discover these trade-offs "
            "efficiently. The default algorithm (NSGA-II) works well for most problems. "
            "You can try others as you get more comfortable."
        )

    with st.expander("Starter tips", expanded=False):
        st.markdown(
            "- Start with **NSGA-II** unless you know you need something else.\n"
            "- Use a built-in template before writing custom code.\n"
            "- Compare only two goals at a time when reading the scatter plot.\n"
            "- Use **Advanced mode** only after the default flow feels clear."
        )

    with st.expander("For Python and CLI users", expanded=False):
        st.markdown(
            "| Command | Description |\n"
            "|---------|-------------|\n"
            "| `vamos quickstart` | Guided wizard for a first run |\n"
            "| `vamos create-problem` | Scaffold a custom problem file |\n"
            "| `vamos summarize` | Table summary of results |\n"
            "| `vamos bench` | Benchmark suite across algorithms |\n"
            "| `vamos tune` | Hyperparameter tuning |\n"
            "| `vamos check` | Verify installation and backends |\n"
            "| `vamos studio` | Launch this dashboard |\n"
            "| `vamos profile` | Performance profiling |\n"
        )
        st.markdown(
            "\n| Goal | Code |\n"
            "|------|------|\n"
            '| Run a benchmark | `optimize("zdt1", algorithm="nsgaii", max_evaluations=5000)` |\n'
            '| Custom problem | `make_problem(fn, n_var=2, n_obj=2, bounds=[(0,1),(0,1)], encoding="real")` |\n'
            '| Compare seeds | `optimize("zdt1", seed=[0, 1, 2, 3, 4])` |\n'
            "| List problems | `from vamos import available_problem_names; available_problem_names()` |\n"
        )

    with st.expander("Keyboard shortcuts (Streamlit defaults)", expanded=False):
        st.markdown(
            "| Key | Action |\n"
            "|-----|--------|\n"
            "| `R` | Rerun the app |\n"
            "| `C` | Clear cache and rerun |\n"
            "| `L` | Toggle light/dark theme |\n"
        )


# ======================================================================
# Main entry point with tabs
# ======================================================================


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Launch VAMOS Studio (Streamlit).")
    parser.add_argument(
        "--study-dir",
        help="Path to a StudyRunner output directory.",
        default="results",
    )
    args, _ = parser.parse_known_args(argv)

    st = _import_streamlit()
    px = _import_plotly()

    st.set_page_config(
        page_title="VAMOS Studio",
        page_icon="https://raw.githubusercontent.com/NicolasRodriguezUribe/VAMOS/main/docs/assets/VAMOS.jpeg",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _inject_accessibility_css(st)

    _render_studio_shell(st)

    tab_welcome, tab_builder, tab_explore = st.tabs(["Welcome", "Problem Builder", "Explore Results"])

    with tab_welcome:
        _render_welcome_tab(st)

    with tab_builder:
        from vamos.ux.studio.problem_builder import render_problem_builder

        render_problem_builder(st, px)

    with tab_explore:
        from vamos.ux.studio.explore_results import render_explore_tab

        render_explore_tab(st, px, Path(args.study_dir))


__all__ = ["main"]


if __name__ == "__main__":
    main()
