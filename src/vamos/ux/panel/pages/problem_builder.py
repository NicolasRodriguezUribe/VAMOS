"""Problem Builder page — define a problem, configure algorithm, run & view results."""

from __future__ import annotations

from typing import Any

import numpy as np
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
from vamos.ux.studio.services import DynamicsCallback, _build_algorithm_config, _run_algorithm, llm_generate_problem_code


class ProblemBuilderState(param.Parameterized):
    """Reactive state for the unified Problem Builder."""

    # ---- Problem Definition ----
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

    # ---- Algorithm Setup ----
    algorithm = param.Selector(
        default="nsgaii",
        objects=["nsgaii", "moead", "spea2", "smsemoa", "nsgaiii", "ibea", "agemoea", "rvea", "smpso"],
        doc="Algorithm for optimization.",
    )
    pop_size = param.Integer(default=100, bounds=(10, 500), doc="Population size.")
    max_gen = param.Integer(default=250, bounds=(10, 1000), doc="Max generations.")
    seed = param.Integer(default=0, bounds=(0, 99999), doc="Random seed.")

    # ---- Operator config ----
    selection = param.Selector(
        default="Tournament",
        objects=["Tournament", "Random", "Boltzmann", "Ranking", "SUS"],
    )
    tournament_size = param.Integer(default=2, bounds=(2, 10))
    use_archive = param.Boolean(default=False)
    archive_mode = param.Selector(default="bounded", objects=["bounded", "unbounded"])
    archive_prune = param.Selector(
        default="crowding",
        objects=["crowding", "hv", "mc_hv", "knn", "maxmin", "ref_dirs"],
    )
    crossover = param.Selector(
        default="SBX",
        objects=["SBX", "BLX-a", "BLX-ab", "Arithmetic", "WholeArithmetic", "Laplace", "Fuzzy", "PCX", "UNDX", "Simplex"],
    )
    crossover_prob = param.Number(default=0.9, bounds=(0.6, 1.0))
    sbx_dist_index = param.Number(default=20.0, bounds=(5.0, 40.0))
    mutation = param.Selector(
        default="Polynomial",
        objects=["Polynomial", "LinkedPolynomial", "NonUniform", "Gaussian", "Cauchy", "Uniform", "UniformReset", "LevyFlight", "PowerLaw"],
    )
    mutation_prob_factor = param.Number(default=1.0, bounds=(0.25, 3.0))
    mutation_dist_index = param.Number(default=20.0, bounds=(5.0, 40.0))

    # ---- AI Assistant ----
    ai_provider = param.Selector(
        default="gemini",
        objects=["gemini", "openai", "anthropic"],
        doc="LLM provider for code generation.",
    )
    ai_api_key = param.String(default="", doc="API key (optional if set via env var).")
    ai_description = param.String(default="", doc="Natural language problem description.")
    ai_status = param.String(default="", precedence=-1)

    # ---- Outputs ----
    preview_plot = param.Parameter(default=None, precedence=-1)
    generated_script = param.String(default="", precedence=-1)
    status_text = param.String(default="", precedence=-1)
    running = param.Boolean(default=False, precedence=-1)
    generation_history = param.List(default=[], precedence=-1)
    current_gen = param.Integer(default=0, bounds=(0, 10000), precedence=-1)
    result_F = param.Array(default=None, precedence=-1)

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

    # ---- Conditional visibility helpers ----

    @pn.depends("selection")
    def tournament_panel(self) -> pn.Column:
        if self.selection == "Tournament":
            return pn.Column(pn.widgets.IntInput.from_param(self.param.tournament_size, name="Tournament size"))
        return pn.Column()

    @pn.depends("use_archive")
    def archive_panel(self) -> pn.Column:
        if self.use_archive:
            widgets = [pn.widgets.Select.from_param(self.param.archive_mode, name="Archive mode")]
            if self.archive_mode == "bounded":
                widgets.append(pn.widgets.Select.from_param(self.param.archive_prune, name="Archive prune"))
            return pn.Column(*widgets)
        return pn.Column()

    @pn.depends("crossover")
    def crossover_params_panel(self) -> pn.Column:
        widgets: list[pn.widgets.Widget] = []
        if self.crossover == "SBX":
            widgets.append(pn.widgets.FloatInput.from_param(self.param.sbx_dist_index, name="SBX dist. index"))
        return pn.Column(*widgets)

    # ---- Generation viewer ----

    @pn.depends("current_gen")
    def generation_view(self) -> pn.pane.Plotly | pn.pane.Markdown:
        if not self.generation_history:
            return pn.pane.Markdown("*No generations recorded yet.*")
        idx = min(self.current_gen, len(self.generation_history) - 1)
        F = self.generation_history[idx]
        import plotly.express as px

        if F.shape[1] == 2:
            fig = px.scatter(x=F[:, 0], y=F[:, 1], labels={"x": "f1", "y": "f2"}, title=f"Generation {idx}")
        elif F.shape[1] >= 3:
            fig = px.scatter_3d(x=F[:, 0], y=F[:, 1], z=F[:, 2], title=f"Generation {idx}")
        else:
            fig = px.scatter(x=F[:, 0], y=np.zeros(len(F)), title=f"Generation {idx}")
        fig.update_layout(height=450)
        return pn.pane.Plotly(fig, sizing_mode="stretch_width")

    # ---- Actions ----

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
                import plotly.express as px

                F_arr = np.asarray(F)
                if F_arr.shape[1] == 2:
                    fig = px.scatter(x=F_arr[:, 0], y=F_arr[:, 1], labels={"x": "f1", "y": "f2"})
                    fig.update_layout(title="Pareto Front Preview")
                elif F_arr.shape[1] == 3:
                    fig = px.scatter_3d(
                        x=F_arr[:, 0],
                        y=F_arr[:, 1],
                        z=F_arr[:, 2],
                        labels={"x": "f1", "y": "f2", "z": "f3"},
                    )
                    fig.update_layout(title="Pareto Front Preview (3D)")
                else:
                    import pandas as pd

                    df = pd.DataFrame(F_arr, columns=[f"f{i + 1}" for i in range(F_arr.shape[1])])
                    fig = px.parallel_coordinates(df)
                    fig.update_layout(title="Pareto Front Preview (Parallel Coordinates)")
                self.preview_plot = fig
                self.status_text = f"Preview complete — {F_arr.shape[0]} solutions found."
            else:
                self.status_text = "Preview returned no objectives."
                self.preview_plot = None
        except Exception as exc:
            self.status_text = f"Error: {exc}"
            self.preview_plot = None

    def run_full_experiment(self, event: Any = None) -> None:
        """Run a full optimization using the registry problem and collect generation history."""
        self.running = True
        self.status_text = "Running full experiment..."
        self.generation_history = []
        self.current_gen = 0
        try:
            from vamos.foundation.problem.registry import make_problem_selection

            selection = make_problem_selection(self.template_name)
            problem = selection.instantiate()
            n_var = getattr(problem, "n_var", None)
            n_obj = getattr(problem, "n_obj", None)

            algo_cfg = _build_algorithm_config(
                self.algorithm,
                pop_size=self.pop_size,
                n_var=n_var,
                n_obj=n_obj,
                encoding=getattr(problem, "encoding", None),
            )
            callback = DynamicsCallback()
            _run_algorithm(
                problem,
                algorithm=self.algorithm,
                algorithm_config=algo_cfg,
                termination=("max_evaluations", self.pop_size * self.max_gen),
                seed=self.seed,
                engine="numpy",
                live_viz=callback,
            )
            self.generation_history = list(callback.history)
            if self.generation_history:
                self.current_gen = len(self.generation_history) - 1
                self.result_F = self.generation_history[-1]
            self.status_text = f"Done — {len(self.generation_history)} generations recorded."
        except Exception as exc:
            self.status_text = f"Error: {exc}"
        finally:
            self.running = False

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

    def ai_generate(self, event: Any = None) -> None:
        """Use an LLM to generate VAMOS code from natural language."""
        if not self.ai_description.strip():
            self.ai_status = "Please enter a problem description."
            return
        self.ai_status = f"Generating with {self.ai_provider}..."
        try:
            result = llm_generate_problem_code(self.ai_description, provider=self.ai_provider, api_key=self.ai_api_key)
            self.objective_code = result["objective_code"]
            self.constraint_code = result.get("constraint_code", "")
            self.n_var = int(result["n_var"])
            self.n_obj = int(result["n_obj"])
            self.bounds_text = str(result["bounds"])
            self.ai_status = "Code generated successfully. Check the Problem Definition tab."
        except Exception as exc:
            self.ai_status = f"Error: {exc}"


def render_problem_builder() -> pn.Column:
    """Build and return the Problem Builder page layout with tabs."""
    state = ProblemBuilderState()

    # ---- Template selector ----
    template_select = pn.widgets.Select.from_param(state.param.template_name, name="Template")

    # ---- Code editor ----
    code_editor = pn.widgets.CodeEditor.from_param(
        state.param.objective_code,
        language="python",
        theme="chrome",
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
        theme="chrome",
        height=100,
        name="Constraint code (optional)",
    )

    # ---- Algorithm config ----
    algo_general = pn.Column(
        pn.widgets.Select.from_param(state.param.algorithm, name="Algorithm"),
        pn.widgets.IntInput.from_param(state.param.pop_size, name="Population size"),
        pn.widgets.IntInput.from_param(state.param.max_gen, name="Max generations"),
        pn.widgets.IntInput.from_param(state.param.seed, name="Seed"),
    )

    # ---- Operator config ----
    operators = pn.Column(
        "### Selection",
        pn.widgets.Select.from_param(state.param.selection, name="Selection"),
        state.tournament_panel,
        pn.layout.Divider(),
        "### Archive",
        pn.widgets.Checkbox.from_param(state.param.use_archive, name="Use external archive"),
        state.archive_panel,
        pn.layout.Divider(),
        "### Crossover",
        pn.widgets.Select.from_param(state.param.crossover, name="Crossover"),
        pn.widgets.FloatInput.from_param(state.param.crossover_prob, name="Crossover probability"),
        state.crossover_params_panel,
        pn.layout.Divider(),
        "### Mutation",
        pn.widgets.Select.from_param(state.param.mutation, name="Mutation"),
        pn.widgets.FloatInput.from_param(state.param.mutation_prob_factor, name="Mutation prob. factor"),
        pn.widgets.FloatInput.from_param(state.param.mutation_dist_index, name="Mutation dist. index"),
    )

    # ---- Action buttons ----
    preview_btn = pn.widgets.Button(name="Quick Preview", button_type="primary")
    preview_btn.on_click(state.run_preview)

    run_btn = pn.widgets.Button(
        name="Run Full Experiment",
        button_type="success",
        disabled=pn.bind(lambda r: r, state.param.running),
    )
    run_btn.on_click(state.run_full_experiment)

    generate_btn = pn.widgets.Button(name="Generate Script", button_type="warning")
    generate_btn.on_click(state.do_generate_script)

    status_pane = pn.pane.Alert(pn.bind(lambda t: t or "Ready.", state.param.status_text), alert_type="info")

    # ---- Preview plot ----
    plot_pane = pn.pane.Plotly(pn.bind(lambda p: p, state.param.preview_plot), sizing_mode="stretch_both", min_height=400)

    # ---- Generation slider + live view ----
    gen_slider = pn.widgets.IntSlider.from_param(
        state.param.current_gen,
        name="Generation",
        start=0,
        end=pn.bind(lambda h: max(len(h) - 1, 0), state.param.generation_history),
    )

    # ---- Generated script ----
    script_pane = pn.pane.Str(
        pn.bind(lambda s: s, state.param.generated_script),
        sizing_mode="stretch_width",
    )

    # ==== Tab 1: AI Assistant ====
    ai_provider_select = pn.widgets.Select.from_param(state.param.ai_provider, name="LLM Provider")
    ai_api_key_input = pn.widgets.PasswordInput.from_param(
        state.param.ai_api_key,
        name="API Key",
        placeholder="Paste your API key here (or set via environment variable)",
    )
    ai_description_input = pn.widgets.TextAreaInput.from_param(
        state.param.ai_description,
        name="Describe your optimization problem",
        placeholder="E.g.: Minimize cost and deflection of a cantilever beam. "
        "Width between 0.5 and 5 cm, height between 1 and 10 cm. "
        "Stress must not exceed 100 MPa.",
        height=150,
    )
    ai_generate_btn = pn.widgets.Button(name="Generate Code", button_type="primary")
    ai_generate_btn.on_click(state.ai_generate)
    ai_status_pane = pn.pane.Alert(
        pn.bind(lambda t: t or "Describe your problem and click Generate Code.", state.param.ai_status),
        alert_type="info",
    )

    tab_ai = pn.Column(
        "### AI-Powered Problem Generation",
        pn.pane.Markdown(
            "Describe your multi-objective optimization problem in natural language. "
            "The AI will generate the objective function code, constraints, variables, "
            "and bounds in VAMOS format. You can review and edit the generated code in "
            "the **Problem Definition** tab.",
        ),
        pn.layout.Divider(),
        pn.Row(ai_provider_select, ai_api_key_input, sizing_mode="stretch_width"),
        ai_description_input,
        ai_generate_btn,
        ai_status_pane,
        sizing_mode="stretch_width",
    )

    # ==== Tab 2: Problem Definition ====
    tab_problem = pn.Column(
        "### Template",
        template_select,
        pn.layout.Divider(),
        "### Objective Function",
        code_editor,
        pn.layout.Divider(),
        "### Variables & Bounds",
        var_controls,
        pn.layout.Divider(),
        "### Constraints (optional)",
        constraint_editor,
        sizing_mode="stretch_width",
    )

    # ==== Tab 3: Algorithm & Operators ====
    tab_algorithm = pn.Column(
        "### General",
        algo_general,
        pn.layout.Divider(),
        operators,
        sizing_mode="stretch_width",
    )

    # ==== Tab 4: Results ====
    tab_results = pn.Column(
        pn.Row(preview_btn, run_btn, generate_btn),
        status_pane,
        pn.layout.Divider(),
        "### Pareto Front Preview",
        plot_pane,
        pn.layout.Divider(),
        "### Generation History",
        state.generation_view,
        gen_slider,
        pn.layout.Divider(),
        "### Generated Script",
        script_pane,
        sizing_mode="stretch_both",
    )

    tabs = pn.Tabs(
        ("AI Assistant", tab_ai),
        ("Problem Definition", tab_problem),
        ("Algorithm & Operators", tab_algorithm),
        ("Results", tab_results),
        sizing_mode="stretch_both",
        dynamic=False,
    )

    return pn.Column(
        "## Problem Builder",
        pn.layout.Divider(),
        tabs,
        sizing_mode="stretch_both",
    )
