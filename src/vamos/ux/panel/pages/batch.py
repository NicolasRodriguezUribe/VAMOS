"""Batch Comparison page — run multiple algorithms x problems and compare."""

from __future__ import annotations

from typing import Any

import numpy as np
import panel as pn
import param


class BatchState(param.Parameterized):
    """Reactive state for batch experiment execution."""

    algorithms = param.ListSelector(
        default=["nsgaii"],
        objects=["nsgaii", "moead", "spea2", "smsemoa", "nsgaiii", "ibea", "agemoea", "rvea", "smpso"],
        doc="Algorithms to compare.",
    )
    problems = param.String(default="zdt1, zdt2, zdt3", doc="Comma-separated problem names.")
    n_runs = param.Integer(default=5, bounds=(1, 30), doc="Independent runs per combination.")
    pop_size = param.Integer(default=100, bounds=(10, 500))
    max_evaluations = param.Integer(default=25000, bounds=(1000, 500000))

    running = param.Boolean(default=False, precedence=-1)
    status = param.String(default="Configure and click Run.", precedence=-1)
    results_df = param.Parameter(default=None, precedence=-1)

    def run_batch(self, event: Any = None) -> None:
        """Execute batch experiments."""
        import pandas as pd

        from vamos.foundation.problem.registry import make_problem_selection
        from vamos.ux.studio.services import _build_algorithm_config, _run_algorithm

        self.running = True
        self.status = "Running batch experiments..."
        problem_list = [p.strip() for p in self.problems.split(",") if p.strip()]
        rows: list[dict[str, Any]] = []

        for prob_name in problem_list:
            try:
                selection = make_problem_selection(prob_name)
                problem = selection.instantiate()
            except Exception as exc:
                self.status = f"Cannot load problem '{prob_name}': {exc}"
                continue

            n_var = getattr(problem, "n_var", None)
            n_obj = getattr(problem, "n_obj", None)
            encoding = getattr(problem, "encoding", None)

            for algo_name in self.algorithms:
                hv_values: list[float] = []
                for run_i in range(self.n_runs):
                    try:
                        cfg = _build_algorithm_config(algo_name, pop_size=self.pop_size, n_var=n_var, n_obj=n_obj, encoding=encoding)
                        result = _run_algorithm(
                            problem,
                            algorithm=algo_name,
                            algorithm_config=cfg,
                            termination=("max_evaluations", self.max_evaluations),
                            seed=run_i,
                            engine="numpy",
                        )
                        F = result.get("F")
                        if F is not None:
                            hv_values.append(float(np.max(np.asarray(F)[:, 0])))
                    except Exception as exc:
                        self.status = f"Run failed for {algo_name} on {prob_name}: {exc}"

                if hv_values:
                    rows.append(
                        {
                            "Problem": prob_name,
                            "Algorithm": algo_name,
                            "Mean": float(np.mean(hv_values)),
                            "Std": float(np.std(hv_values)),
                            "Runs": len(hv_values),
                        }
                    )

        if rows:
            self.results_df = pd.DataFrame(rows)
            self.status = f"Batch complete — {len(rows)} entries."
        else:
            self.status = "No results produced."
        self.running = False

    def export_latex(self, event: Any = None) -> None:
        """Export results table as LaTeX."""
        if self.results_df is None:
            self.status = "No results to export."
            return
        from pathlib import Path

        out = Path("batch_results.tex")
        self.results_df.to_latex(out, index=False, float_format="%.4f")
        self.status = f"LaTeX table saved to {out.resolve()}"

    def export_excel(self, event: Any = None) -> None:
        """Export results table as Excel."""
        if self.results_df is None:
            self.status = "No results to export."
            return
        from pathlib import Path

        out = Path("batch_results.xlsx")
        self.results_df.to_excel(out, index=False)
        self.status = f"Excel file saved to {out.resolve()}"


def render_batch() -> pn.Column:
    """Build and return the Batch Comparison page layout."""
    state = BatchState()

    config = pn.Column(
        "## Batch Comparison",
        pn.widgets.CrossSelector.from_param(state.param.algorithms, name="Algorithms"),
        pn.widgets.TextInput.from_param(state.param.problems, name="Problems (comma-separated)"),
        pn.widgets.IntInput.from_param(state.param.n_runs, name="Runs per combination"),
        pn.widgets.IntInput.from_param(state.param.pop_size, name="Population size"),
        pn.widgets.IntInput.from_param(state.param.max_evaluations, name="Max evaluations"),
    )

    run_btn = pn.widgets.Button(name="Run Batch", button_type="primary")
    run_btn.on_click(state.run_batch)

    latex_btn = pn.widgets.Button(name="Export LaTeX", button_type="warning")
    latex_btn.on_click(state.export_latex)

    excel_btn = pn.widgets.Button(name="Export Excel", button_type="success")
    excel_btn.on_click(state.export_excel)

    status = pn.pane.Alert(pn.bind(lambda s: s, state.param.status), alert_type="info")

    @pn.depends(state.param.results_df)
    def results_table(df: Any) -> pn.widgets.Tabulator | pn.pane.Markdown:
        if df is None:
            return pn.pane.Markdown("*No results yet.*")
        return pn.widgets.Tabulator(df, sizing_mode="stretch_width", show_index=False, theme="fast")

    return pn.Column(
        pn.Row(
            pn.Column(config, run_btn, pn.Row(latex_btn, excel_btn), status, max_width=500),
            pn.Column("## Results", results_table, sizing_mode="stretch_both"),
            sizing_mode="stretch_both",
        ),
        sizing_mode="stretch_both",
    )
