"""Thin entry point for the Studio Streamlit application."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from vamos.ux.studio.app_shell import (
    inject_accessibility_css,
    render_studio_shell,
)
from vamos.ux.studio.app_welcome import (
    _render_first_launch_walkthrough,
    render_welcome_expanders,
    render_welcome_tab,
)

_inject_accessibility_css = inject_accessibility_css
_render_studio_shell = render_studio_shell
_render_welcome_tab = render_welcome_tab
_render_welcome_expanders = render_welcome_expanders


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

    inject_accessibility_css(st)
    render_studio_shell(st)

    tab_welcome, tab_builder, tab_explore = st.tabs(["Welcome", "Problem Builder", "Explore Results"])

    with tab_welcome:
        render_welcome_tab(st)

    with tab_builder:
        from vamos.ux.studio.problem_builder import render_problem_builder

        render_problem_builder(st, px)

    with tab_explore:
        from vamos.ux.studio.explore_results import render_explore_tab

        render_explore_tab(st, px, Path(args.study_dir))


__all__ = ["main"]


if __name__ == "__main__":
    main()
