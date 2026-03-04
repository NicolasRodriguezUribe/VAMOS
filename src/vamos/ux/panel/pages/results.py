"""Results Explorer page — load saved studies, rank with MCDM, export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import panel as pn
import param


class ResultsState(param.Parameterized):
    """Reactive state for the Results Explorer."""

    study_dir = param.String(default="results", doc="Directory containing study results.")
    mcdm_method = param.Selector(
        default="weighted_sum",
        objects=["weighted_sum", "tchebycheff", "knee", "topsis"],
        doc="MCDM ranking method.",
    )
    top_k = param.Integer(default=10, bounds=(1, 100), doc="Number of top solutions to show.")

    # ---- outputs ----
    status = param.String(default="Enter a study directory and click Load.", precedence=-1)
    fronts = param.List(default=[], precedence=-1)
    runs = param.List(default=[], precedence=-1)
    selected_front_idx = param.Integer(default=0, bounds=(0, 100), precedence=-1)

    def load_data(self, event: Any = None) -> None:
        """Load study data from disk."""
        from vamos.ux.studio.services import build_demo_study_data, load_studio_data

        p = Path(self.study_dir)
        try:
            if p.exists() and p.is_dir():
                self.runs, self.fronts = load_studio_data(p)
                self.status = f"Loaded {len(self.runs)} runs, {len(self.fronts)} fronts from {p}."
            else:
                self.runs, self.fronts = build_demo_study_data()
                self.status = f"Directory not found — loaded demo data ({len(self.fronts)} fronts)."
        except Exception as exc:
            self.runs, self.fronts = build_demo_study_data()
            self.status = f"Error loading: {exc}. Using demo data."

    @pn.depends("fronts", "mcdm_method", "top_k")
    def ranking_table(self) -> pn.widgets.Tabulator | pn.pane.Markdown:
        if not self.fronts:
            return pn.pane.Markdown("*Load study data first.*")

        import pandas as pd

        from vamos.ux.studio.dm import build_decision_view, rank_by_score

        rows: list[dict[str, Any]] = []
        for front in self.fronts:
            view = build_decision_view(front, methods=[self.mcdm_method, "weighted_sum", "knee"])
            try:
                order = rank_by_score(view, self.mcdm_method)
            except ValueError:
                order = np.arange(front.points_F.shape[0])
            top_indices = order[: self.top_k]
            for rank_i, idx in enumerate(top_indices):
                row: dict[str, Any] = {
                    "Rank": rank_i + 1,
                    "Algorithm": front.algorithm_name,
                    "Problem": front.problem_name,
                }
                for obj_i in range(front.points_F.shape[1]):
                    row[f"f{obj_i + 1}"] = float(front.points_F[idx, obj_i])
                rows.append(row)

        if not rows:
            return pn.pane.Markdown("*No solutions to rank.*")
        df = pd.DataFrame(rows)
        return pn.widgets.Tabulator(df, sizing_mode="stretch_width", show_index=False, theme="fast")

    @pn.depends("fronts")
    def pareto_overlay(self) -> pn.pane.Plotly | pn.pane.Markdown:
        if not self.fronts:
            return pn.pane.Markdown("*No fronts loaded.*")

        import plotly.graph_objects as go

        fig = go.Figure()
        for front in self.fronts:
            F = front.points_F
            if F.shape[1] >= 2:
                fig.add_trace(
                    go.Scatter(
                        x=F[:, 0],
                        y=F[:, 1],
                        mode="markers",
                        name=front.algorithm_name,
                    )
                )
        fig.update_layout(title="Pareto Front Overlay", xaxis_title="f1", yaxis_title="f2", height=450)
        return pn.pane.Plotly(fig, sizing_mode="stretch_width")

    def export_csv(self, event: Any = None) -> None:
        if not self.fronts:
            self.status = "No data to export."
            return
        from vamos.ux.studio.dm import build_decision_view
        from vamos.ux.studio.export import export_solutions_to_csv

        front = self.fronts[0]
        view = build_decision_view(front)
        out = Path("solutions.csv")
        export_solutions_to_csv(view, list(range(min(self.top_k, front.points_F.shape[0]))), out)
        self.status = f"CSV saved to {out.resolve()}"


def render_results(study_dir: Path | None = None) -> pn.Column:
    """Build and return the Results Explorer page layout."""
    state = ResultsState(study_dir=str(study_dir or "results"))

    load_btn = pn.widgets.Button(name="Load Study", button_type="primary")
    load_btn.on_click(state.load_data)

    export_btn = pn.widgets.Button(name="Export CSV", button_type="success")
    export_btn.on_click(state.export_csv)

    config = pn.Column(
        "## Explore Results",
        pn.widgets.TextInput.from_param(state.param.study_dir, name="Study directory"),
        load_btn,
        pn.layout.Divider(),
        pn.widgets.Select.from_param(state.param.mcdm_method, name="MCDM method"),
        pn.widgets.IntInput.from_param(state.param.top_k, name="Top K solutions"),
        export_btn,
        pn.pane.Alert(pn.bind(lambda s: s, state.param.status), alert_type="info"),
        sizing_mode="stretch_width",
        max_width=400,
    )

    output = pn.Column(
        "## Pareto Fronts",
        state.pareto_overlay,
        pn.layout.Divider(),
        "## Solution Rankings",
        state.ranking_table,
        sizing_mode="stretch_both",
    )

    return pn.Column(
        pn.Row(config, output, sizing_mode="stretch_both"),
        sizing_mode="stretch_both",
    )
