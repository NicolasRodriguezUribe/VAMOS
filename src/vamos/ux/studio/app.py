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
/* --- Dark-mode-aware palette --- */
:root {
    --vamos-accent: #4f8bf9;
    --vamos-bg-card: rgba(255,255,255,0.04);
    --vamos-border: rgba(128,128,128,0.2);
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
        "<h4>First time here? Quick guided tour</h4>"
        "<ol>"
        "<li><b>Problem Builder</b> -- define objectives visually and see the Pareto front live.</li>"
        "<li><b>Explore Results</b> -- load optimization results and compare algorithms.</li>"
        "<li><b>Keyboard shortcuts</b> -- press <code>R</code> to rerun, "
        "<code>C</code> to clear cache.</li>"
        "</ol>"
        '<p style="margin-bottom:0">Pick a tab above to start, '
        "or scroll down for quick-reference tables.</p>"
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
    st.markdown("VAMOS Studio is your interactive workspace for multi-objective optimization. Here is how to get started:")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("1. Build a Problem")
        st.markdown(
            "Go to the **Problem Builder** tab to define your own optimization "
            "problem visually. Write your objective functions, set bounds, "
            "and see the Pareto front update live."
        )
        st.code(
            "from vamos import make_problem\n\n"
            "problem = make_problem(\n"
            "    lambda x: [x[0], 1-x[0]**0.5],\n"
            "    n_var=2, n_obj=2,\n"
            "    bounds=[(0,1), (0,1)],\n"
            ")",
            language="python",
        )

    with col2:
        st.subheader("2. Run Optimization")
        st.markdown("Use the Python API or the CLI to run experiments. Results are stored under `results/` automatically.")
        st.code(
            "# Python\n"
            "from vamos import optimize\n"
            'result = optimize("zdt1",\n'
            '    algorithm="nsgaii",\n'
            "    max_evaluations=10000)\n\n"
            "# Or CLI\n"
            "vamos quickstart",
            language="python",
        )

    with col3:
        st.subheader("3. Explore Results")
        st.markdown(
            "Switch to the **Explore Results** tab to compare algorithms, rank solutions with MCDM methods, and export the best ones."
        )
        st.code(
            "vamos summarize \\\n  --results results/\n\n# Or browse visually in\n# the Explore Results tab",
            language="bash",
        )

    st.divider()
    _render_welcome_expanders(st)


def _render_welcome_expanders(st: Any) -> None:
    """Render the collapsible quick-reference sections."""
    with st.expander("Quick Reference: CLI commands", expanded=False):
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

    with st.expander("Quick Reference: Python API", expanded=False):
        st.markdown(
            "| Goal | Code |\n"
            "|------|------|\n"
            '| Run a benchmark | `optimize("zdt1", algorithm="nsgaii", max_evaluations=5000)` |\n'
            "| Custom problem | `make_problem(fn, n_var=2, n_obj=2, bounds=[(0,1),(0,1)])` |\n"
            '| Compare seeds | `optimize("zdt1", seed=[0, 1, 2, 3, 4])` |\n'
            "| List problems | `from vamos import available_problem_names; available_problem_names()` |\n"
        )

    with st.expander("What is multi-objective optimization?", expanded=False):
        st.markdown(
            "Multi-objective optimization finds solutions that balance "
            "**conflicting objectives** (e.g. cost vs quality). Instead of a "
            "single best answer, you get a **Pareto front** -- a set of "
            "trade-off solutions where improving one objective worsens another.\n\n"
            "VAMOS uses **evolutionary algorithms** (MOEAs) to discover these "
            "fronts efficiently. Popular algorithms include NSGA-II, MOEA/D, "
            "and SPEA2."
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

    st.title("VAMOS Studio")

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
