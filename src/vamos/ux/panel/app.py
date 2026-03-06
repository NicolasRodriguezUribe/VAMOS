"""Main entry point for the VAMOS Studio Panel application."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import panel as pn
import param

from vamos.ux.panel.shell import PAGES, build_sidebar_nav, build_template


def _ensure_panel_extension() -> None:
    pn.extension("codeeditor", "plotly", "tabulator", sizing_mode="stretch_width")


class StudioApp(param.Parameterized):
    """Top-level application state coordinating page navigation."""

    page = param.Selector(
        default="welcome",
        objects=list(PAGES.keys()),
        doc="Currently active page slug.",
    )
    study_dir = param.String(default="results", doc="Root directory for study results.")

    # ---- internal ----
    _nav_col = param.ClassSelector(class_=pn.Column, precedence=-1)
    _main_area = param.ClassSelector(class_=pn.Column, precedence=-1)

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._nav_col = build_sidebar_nav(self.page)
        self._main_area = pn.Column(sizing_mode="stretch_both")
        self._wire_nav_buttons()
        self._refresh_main()

    # ---- navigation wiring ----

    def _wire_nav_buttons(self) -> None:
        for btn in self._nav_col:
            if isinstance(btn, pn.widgets.Button):
                btn.on_click(self._on_nav_click)

    def _on_nav_click(self, event: Any) -> None:
        slug = event.obj.tags[0] if event.obj.tags else "welcome"
        self.page = slug

    @param.depends("page", watch=True)
    def _refresh_main(self) -> None:
        # Update sidebar button styles
        for btn in self._nav_col:
            if isinstance(btn, pn.widgets.Button):
                slug = btn.tags[0] if btn.tags else ""
                btn.button_type = "primary" if slug == self.page else "light"

        # Lazy-load the selected page
        self._main_area.clear()
        self._main_area.append(self._load_page(self.page))

    def _load_page(self, slug: str) -> pn.viewable.Viewable:
        if slug == "welcome":
            from vamos.ux.panel.pages.welcome import render_welcome

            return render_welcome()
        if slug == "problem_builder":
            from vamos.ux.panel.pages.problem_builder import render_problem_builder

            return render_problem_builder()
        if slug == "batch":
            from vamos.ux.panel.pages.batch import render_batch

            return render_batch()
        if slug == "results":
            from vamos.ux.panel.pages.results import render_results

            return render_results(study_dir=Path(self.study_dir))
        return pn.pane.Markdown(f"## Unknown page: {slug}")

    # ---- public: composable template ----

    def view(self) -> pn.template.FastListTemplate:
        tmpl = build_template()
        tmpl.sidebar.append(
            pn.pane.Markdown(
                "### VAMOS Studio",
                align="center",
                styles={"text-align": "center", "padding": "0.5rem 0"},
            )
        )
        tmpl.sidebar.append(pn.layout.Divider())
        tmpl.sidebar.append(self._nav_col)
        tmpl.main.append(self._main_area)
        return tmpl


# ---------------------------------------------------------------------------
# Servable entry point (used by `panel serve app.py`)
# ---------------------------------------------------------------------------


def create_app() -> pn.template.FastListTemplate:
    """Factory called when Panel serves this module."""
    _ensure_panel_extension()
    app = StudioApp()
    return app.view()
