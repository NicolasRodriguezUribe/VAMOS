"""VAMOS Studio -- main Streamlit application.

Thin entry point that orchestrates three tabs:

1. **Welcome** -- onboarding wizard
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
# Tab: Welcome / Onboarding
# ======================================================================


def _render_welcome_tab(st: Any) -> None:
    """Render the Welcome / Getting Started onboarding page."""
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
