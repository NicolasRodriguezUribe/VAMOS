"""Problem Builder page — define or pick a template problem and preview results."""

from __future__ import annotations

from typing import Any

import panel as pn
import param

from vamos.ux.studio.problem_builder_backend import (
    compile_constraint_function,
    compile_objective_function,
    example_objectives,
    generate_script,
    parse_bounds_text,
    run_preview_optimization,
)
from vamos.ux.studio.problem_builder_templates import _DEFAULT_TEMPLATE


class ProblemBuilderState(param.Parameterized):
    """Reactive state for the Problem Builder."""

    template_name = param.Selector(
        default=_DEFAULT_TEMPLATE,
        objects=list(example_objectives().keys()),
        doc="Selected problem template.",
    )
    objective_code = param.String(default="", doc="Python code for objectives.")
    n_var = param.Integer(default=5, bounds=(1, 100), doc="Number of decision variables.")
    n_obj = param.Integer(default=2, bounds=(2, 10), doc="Number of objectives.")
    bounds_text = param.String(default="0.0, 1.0", doc="Bounds: one line per var, or single line for all.")
    constraint_code = param.String(default="", doc="Optional constraint code.")
    algorithm = param.Selector(
        default="nsgaii",
        objects=["nsgaii", "moead", "spea2", "smsemoa", "nsgaiii", "ibea", "agemoea", "rvea", "smpso"],
        doc="Algorithm for preview.",
    )
    pop_size = param.Integer(default=50, bounds=(10, 500), doc="Population size for preview.")
    max_gen = param.Integer(default=100, bounds=(10, 1000), doc="Max generations for preview.")
    seed = param.Integer(default=42, bounds=(0, 99999), doc="Random seed.")

    # Outputs
    preview_plot = param.Parameter(default=None, precedence=-1)
    generated_script = param.String(default="", precedence=-1)
    status_text = param.String(default="", precedence=-1)

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._apply_template()

    @param.depends("template_name", watch=True)
    def _apply_template(self) -> None:
        templates = example_objectives()
        t = templates.get(self.template_name, {})
        self.objective_code = t.get("code", "")
        self.n_var = int(t.get("n_var", 2))
        self.n_obj = int(t.get("n_obj", 2))
        self.bounds_text = "0.0, 1.0"

    def _parse_bounds(self) -> list[tuple[float, float]] | None:
        result = parse_bounds_text(self.bounds_text, self.n_var)
        if isinstance(result, str):
            self.status_text = f"Bounds error: {result}"
            return None
        return result

    def run_preview(self, event: Any = None) -> None:
        """Compile code, run a short optimization, update preview_plot."""
        self.status_text = "Compiling..."
        try:
            fn = compile_objective_function(self.objective_code)
        except Exception as exc:
            self.status_text = f"Compilation error: {exc}"
            self.preview_plot = None
            return

        bounds = self._parse_bounds()
        if bounds is None:
            return

        constraints = None
        n_constraints = 0
        if self.constraint_code.strip():
            try:
                constraints = compile_constraint_function(self.constraint_code)
                n_constraints = 1
            except Exception as exc:
                self.status_text = f"Constraint error: {exc}"
                return

        self.status_text = "Running preview..."
        try:
            result = run_preview_optimization(
                fn,
                n_var=self.n_var,
                n_obj=self.n_obj,
                bounds=bounds,
                algorithm=self.algorithm,
                pop_size=self.pop_size,
                budget=self.pop_size * self.max_gen,
                seed=self.seed,
                constraints=constraints,
                n_constraints=n_constraints,
                objective_code=self.objective_code,
                constraint_code=self.constraint_code,
            )
            F = result.get("F")
            if F is not None:
                import numpy as np
                import plotly.express as px

                F_arr = np.asarray(F)
                if F_arr.shape[1] == 2:
                    fig = px.scatter(x=F_arr[:, 0], y=F_arr[:, 1], labels={"x": "f1", "y": "f2"})
                    fig.update_layout(title="Pareto Front Preview")
                elif F_arr.shape[1] == 3:
                    fig = px.scatter_3d(
                        x=F_arr[:, 0], y=F_arr[:, 1], z=F_arr[:, 2],
                        labels={"x": "f1", "y": "f2", "z": "f3"},
                    )
                    fig.update_layout(title="Pareto Front Preview (3D)")
                else:
                    import pandas as pd

                    df = pd.DataFrame(F_arr, columns=[f"f{i+1}" for i in range(F_arr.shape[1])])
                    fig = px.parallel_coordinates(df)
                    fig.update_layout(title="Pareto Front Preview (Parallel Coordinates)")
                self.preview_plot = fig
                self.status_text = f"Preview complete \u2014 {F_arr.shape[0]} solutions found."
            else:
                self.status_text = "Preview returned no objectives."
                self.preview_plot = None
        except Exception as exc:
            self.status_text = f"Error: {exc}"
            self.preview_plot = None

    def do_generate_script(self, event: Any = None) -> None:
        """Generate a standalone Python script for the configured problem."""
        bounds = self._parse_bounds()
        if bounds is None:
            return
        try:
            self.generated_script = generate_script(
                self.objective_code,
                name=self.template_name,
                n_var=self.n_var,
                n_obj=self.n_obj,
                bounds=bounds,
                algorithm=self.algorithm,
                budget=self.pop_size * self.max_gen,
                constraint_code=self.constraint_code,
            )
            self.status_text = "Script generated."
        except Exception as exc:
            self.status_text = f"Script generation error: {exc}"


def render_problem_builder() -> pn.Column:
    """Build and return the Problem Builder page layout."""
    state = ProblemBuilderState()

    # ---- Template selector ----
    template_select = pn.widgets.Select.from_param(state.param.template_name, name="Template")

    # ---- Code editor ----
    code_editor = pn.widgets.CodeEditor.from_param(
        state.param.objective_code,
        language="python",
        theme="monokai",
        height=200,
        name="Objective code",
    )

    # ---- Variable config ----
    var_controls = pn.Column(
        pn.widgets.IntInput.from_param(state.param.n_var, name="Variables (n_var)"),
        pn.widgets.IntInput.from_param(state.param.n_obj, name="Objectives (n_obj)"),
        pn.widgets.TextAreaInput.from_param(state.param.bounds_text, name="Bounds (lo, hi per line)"),
    )

    # ---- Constraints ----
    constraint_editor = pn.widgets.CodeEditor.from_param(
        state.param.constraint_code,
        language="python",
        theme="monokai",
        height=100,
        name="Constraint code (optional)",
    )

    # ---- Algorithm config ----
    algo_controls = pn.Column(
        pn.widgets.Select.from_param(state.param.algorithm, name="Algorithm"),
        pn.widgets.IntInput.from_param(state.param.pop_size, name="Population size"),
        pn.widgets.IntInput.from_param(state.param.max_gen, name="Max generations"),
        pn.widgets.IntInput.from_param(state.param.seed, name="Seed"),
    )

    # ---- Action buttons ----
    preview_btn = pn.widgets.Button(name="Run Preview", button_type="primary")
    preview_btn.on_click(state.run_preview)

    generate_btn = pn.widgets.Button(name="Generate Script", button_type="success")
    generate_btn.on_click(state.do_generate_script)

    status_pane = pn.pane.Alert(pn.bind(lambda t: t or "Ready.", state.param.status_text), alert_type="info")

    # ---- Preview plot ----
    plot_pane = pn.pane.Plotly(pn.bind(lambda p: p, state.param.preview_plot), sizing_mode="stretch_both", min_height=400)

    # ---- Generated script ----
    script_pane = pn.pane.Str(
        pn.bind(lambda s: s, state.param.generated_script),
        sizing_mode="stretch_width",
    )

    # ---- Assemble ----
    config_panel = pn.Column(
        "## Problem Builder",
        template_select,
        pn.layout.Divider(),
        "### Objectives",
        code_editor,
        pn.layout.Divider(),
        "### Variables & Bounds",
        var_controls,
        pn.layout.Divider(),
        "### Constraints",
        constraint_editor,
        pn.layout.Divider(),
        "### Algorithm & Preview",
        algo_controls,
        pn.Row(preview_btn, generate_btn),
        status_pane,
        sizing_mode="stretch_width",
        max_width=500,
    )

    output_panel = pn.Column(
        "## Preview",
        plot_pane,
        pn.layout.Divider(),
        "### Generated Script",
        script_pane,
        sizing_mode="stretch_both",
    )

    return pn.Column(
        pn.Row(config_panel, output_panel, sizing_mode="stretch_both"),
        sizing_mode="stretch_both",
    )
