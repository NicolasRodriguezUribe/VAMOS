"""Experiment page — run a single algorithm with live monitoring."""

from __future__ import annotations

from typing import Any

import numpy as np
import panel as pn
import param

from vamos.ux.studio.services import DynamicsCallback, _build_algorithm_config, _run_algorithm


class ExperimentState(param.Parameterized):
    """Reactive state for single-experiment execution and live monitoring."""

    # ---- General ----
    problem_name = param.String(default="zdt1", doc="Problem identifier (registry name or custom).")
    algorithm = param.Selector(
        default="nsgaii",
        objects=["nsgaii", "moead", "spea2", "smsemoa", "nsgaiii", "ibea", "agemoea", "rvea", "smpso"],
    )
    pop_size = param.Integer(default=100, bounds=(10, 500))
    max_evaluations = param.Integer(default=10000, bounds=(100, 500000))
    seed = param.Integer(default=0, bounds=(0, 99999))

    # ---- NSGA-II conditional params ----
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

    # ---- Outputs ----
    running = param.Boolean(default=False, precedence=-1)
    status = param.String(default="Ready.", precedence=-1)
    generation_history = param.List(default=[], precedence=-1)
    current_gen = param.Integer(default=0, bounds=(0, 10000), precedence=-1)
    result_F = param.Array(default=None, precedence=-1)

    # ---- conditional visibility helpers ----

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

    # ---- actions ----

    def run_experiment(self, event: Any = None) -> None:
        """Execute the optimization and collect generation history."""
        self.running = True
        self.status = "Running..."
        self.generation_history = []
        self.current_gen = 0
        try:
            from vamos.foundation.problem.registry import make_problem_selection

            selection = make_problem_selection(self.problem_name)
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
                termination=("max_evaluations", self.max_evaluations),
                seed=self.seed,
                engine="numpy",
                live_viz=callback,
            )
            self.generation_history = list(callback.history)
            if self.generation_history:
                self.current_gen = len(self.generation_history) - 1
                self.result_F = self.generation_history[-1]
            self.status = f"Done — {len(self.generation_history)} generations recorded."
        except Exception as exc:
            self.status = f"Error: {exc}"
        finally:
            self.running = False


def render_experiment() -> pn.Column:
    """Build and return the Experiment page layout."""
    state = ExperimentState()

    # ---- General config ----
    general = pn.Column(
        "### General",
        pn.widgets.TextInput.from_param(state.param.problem_name, name="Problem"),
        pn.widgets.Select.from_param(state.param.algorithm, name="Algorithm"),
        pn.widgets.IntInput.from_param(state.param.pop_size, name="Population size"),
        pn.widgets.IntInput.from_param(state.param.max_evaluations, name="Max evaluations"),
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

    # ---- Run button ----
    run_btn = pn.widgets.Button(name="Run Experiment", button_type="primary", disabled=pn.bind(lambda r: r, state.param.running))
    run_btn.on_click(state.run_experiment)

    status_pane = pn.pane.Alert(pn.bind(lambda s: s, state.param.status), alert_type="info")

    # ---- Generation slider + live view ----
    gen_slider = pn.widgets.IntSlider.from_param(
        state.param.current_gen,
        name="Generation",
        start=0,
        end=pn.bind(lambda h: max(len(h) - 1, 0), state.param.generation_history),
    )

    # ---- Assemble ----
    config_col = pn.Column(
        "## Run Experiment",
        general,
        pn.layout.Divider(),
        operators,
        pn.layout.Divider(),
        run_btn,
        status_pane,
        sizing_mode="stretch_width",
        max_width=420,
    )

    viz_col = pn.Column(
        "## Visualization",
        state.generation_view,
        gen_slider,
        sizing_mode="stretch_both",
    )

    return pn.Column(
        pn.Row(config_col, viz_col, sizing_mode="stretch_both"),
        sizing_mode="stretch_both",
    )
