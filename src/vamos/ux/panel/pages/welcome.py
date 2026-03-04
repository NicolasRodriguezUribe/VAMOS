"""Welcome page for VAMOS Studio."""

from __future__ import annotations

import panel as pn


def render_welcome() -> pn.Column:
    """Render the welcome / onboarding page."""
    hero = pn.pane.Markdown(
        """
# Welcome to VAMOS Studio

**Vectorized Architecture for Multi-Objective Optimization Studies**

VAMOS Studio is your interactive workbench for designing, running and
analysing multi-objective optimization experiments.

---

### Quick start

1. **Problem Builder** — define or pick a template problem, preview the
   Pareto front, and generate a ready-to-run Python script.
2. **Run Experiment** — configure an algorithm (NSGA-II, MOEA/D, SMPSO, ...)
   with full parameter control and watch the optimization live.
3. **Batch Comparison** — run multiple algorithms on multiple problems in
   parallel and produce publication-ready comparison tables.
4. **Explore Results** — load any saved study, rank solutions with MCDM
   methods, and export to CSV / LaTeX / Excel.

Use the sidebar on the left to navigate between pages.
        """,
        sizing_mode="stretch_width",
    )

    tips = pn.pane.Alert(
        "**Tip:** Install the full studio extras with "
        "`pip install -e \".[studio]\"` for interactive charts and tables.",
        alert_type="info",
    )

    return pn.Column(hero, tips, sizing_mode="stretch_both")
