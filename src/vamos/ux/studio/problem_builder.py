"""VAMOS Studio -- Problem Builder page (Streamlit UI layer).

Progressive disclosure design: beginners see templates and a one-click run
button; intermediate users can edit code and tweak algorithms; advanced users
access bounds, constraints, population size and seed.

Delegates heavy lifting to ``problem_builder_backend``.
"""

from __future__ import annotations

from typing import Any

from vamos.ux.studio.problem_builder_backend import (
    _DEFAULT_TEMPLATE,
    compile_constraint_function,
    compile_objective_function,
    example_objectives,
    generate_script,
    parse_bounds_text,
    run_preview_optimization,
)

# Re-export backend symbols so existing tests keep working
_example_objectives = example_objectives
_parse_bounds_text = parse_bounds_text
_generate_script = generate_script


# ------------------------------------------------------------------
# Preview plot rendering
# ------------------------------------------------------------------


def _render_preview_plot(st: Any, px: Any, F: Any, n_obj: int, problem_name: str) -> None:
    """Render the appropriate preview chart for the given objective count."""
    if n_obj == 2:
        import pandas as pd

        df = pd.DataFrame({"f0": F[:, 0], "f1": F[:, 1]})
        fig = px.scatter(
            df,
            x="f0",
            y="f1",
            title=f"Pareto Front Preview -- {problem_name}",
            labels={"f0": "Objective 0 (minimize)", "f1": "Objective 1 (minimize)"},
        )
        fig.update_traces(marker=dict(size=7, opacity=0.8))
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)
    elif n_obj == 3:
        import plotly.graph_objects as go

        fig3d = go.Figure(
            data=[
                go.Scatter3d(
                    x=F[:, 0],
                    y=F[:, 1],
                    z=F[:, 2],
                    mode="markers",
                    marker=dict(size=4, opacity=0.8, color=F[:, 0], colorscale="Viridis"),
                )
            ]
        )
        fig3d.update_layout(
            title=f"Pareto Front Preview -- {problem_name}",
            scene=dict(xaxis_title="f0", yaxis_title="f1", zaxis_title="f2"),
            height=520,
        )
        st.plotly_chart(fig3d, use_container_width=True)
    else:
        import pandas as pd

        cols = [f"f{i}" for i in range(n_obj)]
        df_pc = pd.DataFrame(F, columns=cols)
        fig_pc = px.parallel_coordinates(df_pc, dimensions=cols, title="Pareto Front (parallel coordinates)")
        fig_pc.update_layout(height=420)
        st.plotly_chart(fig_pc, use_container_width=True)


def _render_summary_table(st: Any, F: Any, n_obj: int) -> None:
    """Show a summary stats table for the preview run."""
    import pandas as pd

    rows = []
    for i in range(n_obj):
        rows.append(
            {
                "Objective": f"f{i}",
                "Min": f"{F[:, i].min():.6f}",
                "Max": f"{F[:, i].max():.6f}",
                "Mean": f"{F[:, i].mean():.6f}",
                "Std": f"{F[:, i].std():.6f}",
            }
        )
    st.dataframe(pd.DataFrame(rows).set_index("Objective"), use_container_width=True)


def _coerce_bool_widget_value(value: Any, default: bool = False) -> bool:
    """Normalize Streamlit widget output for tests and mocked render calls."""
    return value if isinstance(value, bool) else default


def _render_template_snapshot(st: Any, template: dict[str, str]) -> None:
    """Show the most important template facts without exposing code editors."""
    constraints = int(template.get("n_constraints", "0"))
    col_var, col_obj, col_con = st.columns(3)
    with col_var:
        st.metric("Variables", template["n_var"])
    with col_obj:
        st.metric("Objectives", template["n_obj"])
    with col_con:
        st.metric("Constraints", str(constraints))


def _render_preview_guidance(st: Any, n_obj: int) -> None:
    """Explain how to read the preview in plain language."""
    if n_obj == 2:
        st.info(
            "How to read this chart: each point is one solution. Points farther down and left are usually better, "
            "but no single point wins on every goal."
        )
    else:
        st.info(
            "How to read this preview: Studio is showing a compact view of several goals at once. "
            "Use the summary table to compare the spread of objective values."
        )


def _render_template_chips(st: Any, template: dict[str, str], *, advanced_mode: bool) -> None:
    """Show the currently selected template mode and difficulty in a compact strip."""
    difficulty = template.get("difficulty", "custom").title()
    category = template.get("category", "general").replace("_", " ").title()
    mode_label = "Advanced mode enabled" if advanced_mode else "Beginner mode enabled"
    st.markdown(
        '<div class="studio-chip-row">'
        f'<span class="studio-chip">{mode_label}</span>'
        f'<span class="studio-chip">{category} template</span>'
        f'<span class="studio-chip">{difficulty} difficulty</span>'
        '<span class="studio-chip">NSGA-II defaults ready</span>'
        "</div>",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# Constraint builder UI
# ------------------------------------------------------------------


def _render_constraint_section(st: Any, template: dict[str, str]) -> tuple[str, int, Any, str | None]:
    """Render the constraint builder and return (code, n_constraints, fn, error)."""
    with st.expander("Constraints (optional)", expanded=bool(template.get("constraint_code"))):
        st.caption("Constraints use the convention **g(x) <= 0 is feasible**. Return a list of constraint values.")
        default_g = template.get("constraint_code", "")
        default_n = int(template.get("n_constraints", "0")) if default_g else 0

        n_constraints = st.number_input(
            "Number of constraints",
            min_value=0,
            max_value=20,
            value=default_n,
            step=1,
            help="How many inequality constraints (g(x) <= 0)?",
        )

        constraint_code = ""
        constraint_fn: Any = None
        constraint_error: str | None = None

        if int(n_constraints) > 0:
            constraint_code = st.text_area(
                "Constraint code",
                value=default_g,
                height=120,
                help="Return a list of constraint values. g(x) <= 0 means feasible.",
            )
            if constraint_code.strip():
                try:
                    constraint_fn = compile_constraint_function(constraint_code)
                except SyntaxError as exc:
                    constraint_error = f"Syntax error on line {exc.lineno}: {exc.msg}"
                except Exception as exc:
                    constraint_error = str(exc)
                if constraint_error:
                    st.error(f"Constraint error: {constraint_error}")

    return constraint_code, int(n_constraints), constraint_fn, constraint_error


# ------------------------------------------------------------------
# Algorithm labels (friendly for all levels)
# ------------------------------------------------------------------

_ALGO_LABELS = {
    "nsgaii": "NSGA-II (Recommended)",
    "moead": "MOEA/D",
    "spea2": "SPEA2",
    "smsemoa": "SMS-EMOA",
    "nsgaiii": "NSGA-III",
    "ibea": "IBEA",
    "agemoea": "AGE-MOEA",
    "rvea": "RVEA",
}

_ALGO_HELP = {
    "nsgaii": "General-purpose, works well on most problems",
    "moead": "Decomposes the problem into scalar sub-problems",
    "spea2": "Maintains strong diversity among solutions",
    "smsemoa": "Driven by hypervolume indicator quality",
    "nsgaiii": "Designed for problems with 3+ objectives",
    "ibea": "Indicator-based evolutionary algorithm",
    "agemoea": "Uses geometry to estimate the Pareto front shape",
    "rvea": "Guided by reference vectors for many objectives",
}


# ------------------------------------------------------------------
# Main Streamlit entry
# ------------------------------------------------------------------


def render_problem_builder(st: Any, px: Any) -> None:
    """Render the Problem Builder tab inside VAMOS Studio."""
    st.header("Problem Builder")
    st.markdown(
        '<div class="studio-section-intro">'
        '<span class="studio-kicker">Builder</span>'
        "<h3>Shape a problem before you worry about optimization jargon</h3>"
        "<p>Start with a ready-made template, inspect its defaults, and only open the deeper controls if you need to customize the mechanics.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Step 1: choose a template. Step 2: run a preview. Step 3: read the trade-off chart. "
        "Code editors stay hidden until you switch on Advanced mode."
    )
    default_advanced = bool(st.session_state.get("studio_advanced_mode", False))
    toggle_widget = getattr(st, "toggle", None)
    if callable(toggle_widget):
        advanced_mode = _coerce_bool_widget_value(
            toggle_widget(
                "Advanced mode",
                value=default_advanced,
                help="Show code editors, algorithm selection, and expert settings.",
            ),
            default=default_advanced,
        )
    else:
        advanced_mode = _coerce_bool_widget_value(
            st.checkbox(
                "Advanced mode",
                value=default_advanced,
                help="Show code editors, algorithm selection, and expert settings.",
            ),
            default=default_advanced,
        )
    st.session_state["studio_advanced_mode"] = advanced_mode

    if not advanced_mode:
        st.info("Beginner mode is active. Studio will use the template defaults and the recommended NSGA-II setup.")

    st.markdown("### Step 1. Choose a starting point")
    # ---- template selector ----
    examples = example_objectives()
    col_cat, col_tmpl = st.columns([2, 2])
    with col_cat:
        categories = sorted({t.get("category", "other") for t in examples.values()})
        cat_labels = {
            "math": "Math benchmarks",
            "engineering": "Engineering",
            "ml": "Machine Learning",
            "scheduling": "Scheduling",
            "blank": "Blank",
        }
        selected_cat = st.selectbox(
            "Domain",
            ["all"] + categories,
            format_func=lambda c: "All templates" if c == "all" else cat_labels.get(c, c.title()),
            help="Filter templates by application domain.",
        )
    filtered = {k: v for k, v in examples.items() if selected_cat == "all" or v.get("category") == selected_cat}
    with col_tmpl:
        default_idx = list(filtered.keys()).index(_DEFAULT_TEMPLATE) if _DEFAULT_TEMPLATE in filtered else 0
        template_name = st.selectbox(
            "Start from a template",
            list(filtered.keys()),
            index=default_idx,
            help="Pick a starting point. You can edit everything below.",
        )
    template = filtered[template_name]

    # ---- template description (visible to all users) ----
    description = template.get("description", "")
    difficulty = template.get("difficulty", "")
    if description:
        difficulty_badge = ""
        if difficulty:
            colors = {"beginner": "green", "intermediate": "orange", "advanced": "red"}
            color = colors.get(difficulty, "gray")
            difficulty_badge = f' <span style="background:{color};color:white;padding:2px 8px;border-radius:10px;font-size:0.75rem;margin-left:8px">{difficulty.title()}</span>'
        st.markdown(f"{description}{difficulty_badge}", unsafe_allow_html=True)
    _render_template_chips(st, template, advanced_mode=advanced_mode)
    _render_template_snapshot(st, template)

    st.markdown("### Step 2. Run a quick preview")
    st.markdown(
        '<div class="studio-data-strip">'
        "<p>Preview runs use a small, fast budget so you can learn the shape of the trade-off before committing to a larger experiment.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    # ---- Quick run with defaults (one-click for beginners) ----
    quick_run = st.button("Run preview", type="primary", use_container_width=True)

    # ---- Compile template defaults (needed for both quick run and custom run) ----
    default_bounds = template.get("bounds", "0.0, 1.0")
    code = template["code"]
    n_var = int(template["n_var"])
    n_obj = int(template["n_obj"])
    algorithm = "nsgaii"
    budget = 2000
    pop_size = 50
    seed = 42
    problem_name = "my_problem"
    constraint_code = ""
    n_constraints = 0
    constraint_fn: Any = None
    constraint_error: str | None = None

    # ---- Customization sections (progressive disclosure) ----
    if advanced_mode:
        # Problem definition
        with st.expander("Edit problem definition", expanded=False):
            problem_name = st.text_input(
                "Problem name",
                value="my_problem",
                help="Human-readable label for logs and exported scripts.",
            )
            col_nvar, col_nobj = st.columns(2)
            with col_nvar:
                n_var = st.number_input(
                    "Decision variables (n_var)",
                    min_value=1,
                    max_value=100,
                    value=int(template["n_var"]),
                    step=1,
                    help="How many inputs does your problem have?",
                )
            with col_nobj:
                n_obj = st.number_input(
                    "Objectives (n_obj)",
                    min_value=2,
                    max_value=10,
                    value=int(template["n_obj"]),
                    step=1,
                    help="How many objectives to minimize (must be >= 2).",
                )

            # Variable bounds
            bounds_text = st.text_area(
                "Variable bounds",
                value=default_bounds,
                height=68,
                help="One line applies to ALL variables. Or one 'lower, upper' per variable.",
            )
            default_bounds = bounds_text

            st.markdown("**Objective function** (receives `x`, return a list of objectives)")
            st.caption("Expected format: compute objective values and `return [f0, f1, ...]` with exactly `n_obj` entries.")
            code = st.text_area(
                "Objective code",
                value=template["code"],
                height=280,
                help="Write Python code that uses `x` and returns a list of n_obj values.",
                label_visibility="collapsed",
            )

            # Constraint builder
            constraint_code, n_constraints, constraint_fn, constraint_error = _render_constraint_section(st, template)

        # Algorithm selection
        with st.expander("Choose algorithm", expanded=False):
            algo_help_text = "\n".join(f"- **{v}**: {_ALGO_HELP[k]}" for k, v in _ALGO_LABELS.items())
            st.markdown(algo_help_text)
            algorithm = str(
                st.selectbox(
                    "Algorithm",
                    list(_ALGO_LABELS.keys()),
                    index=0,
                    format_func=lambda key: _ALGO_LABELS.get(key, key),
                    help="Pick a MOEA. Start with NSGA-II unless you have a specific reason.",
                )
            )

        # Advanced settings
        with st.expander("Advanced settings", expanded=False):
            col_budget, col_pop = st.columns(2)
            with col_budget:
                budget = int(
                    st.number_input(
                        "Max evaluations",
                        min_value=200,
                        max_value=50000,
                        value=2000,
                        step=500,
                        help="Higher = better results, slower preview.",
                    )
                )
            with col_pop:
                pop_size = int(
                    st.number_input(
                        "Population size",
                        min_value=10,
                        max_value=500,
                        value=50,
                        step=10,
                        help="Number of candidate solutions per generation.",
                    )
                )
            seed = int(st.number_input("Seed", min_value=0, value=42, step=1, help="Random seed for reproducibility."))
    else:
        st.caption("Using template defaults. Turn on Advanced mode to edit code, constraints, algorithm, or optimization settings.")

    # ---- Validation ----
    bounds_result = parse_bounds_text(default_bounds, int(n_var))
    if isinstance(bounds_result, str):
        st.error(f"Bounds error: {bounds_result}")
        bounds_ok: list[tuple[float, float]] = []
    else:
        bounds_ok = bounds_result

    compile_error: str | None = None
    fn: Any = None
    if code.strip():
        try:
            fn = compile_objective_function(code)
        except SyntaxError as exc:
            compile_error = f"Syntax error on line {exc.lineno}: {exc.msg}"
        except Exception as exc:
            compile_error = str(exc)
    if compile_error:
        st.error(f"Code error: {compile_error}")

    has_constraint_error = constraint_error is not None

    # ---- Live preview ----
    st.markdown("### Step 3. Read the result")
    if not quick_run:
        st.markdown(
            '<div class="studio-data-strip">'
            "<p>Your preview chart and summary table will appear here after you click <strong>Run preview</strong>.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    if quick_run and fn is not None and bounds_ok and not compile_error and not has_constraint_error:
        _run_and_show_preview(
            st,
            px,
            fn=fn,
            objective_code=code,
            constraint_code=constraint_code,
            problem_name=problem_name,
            n_var=int(n_var),
            n_obj=int(n_obj),
            bounds_ok=bounds_ok,
            algorithm=algorithm,
            budget=budget,
            pop_size=pop_size,
            seed=seed,
            constraints=constraint_fn,
            n_constraints=n_constraints,
        )

    # ---- export ----
    st.divider()
    st.subheader("Export")
    st.markdown(
        '<div class="studio-data-strip">'
        "<p>When the template or custom code is valid, Studio can turn the current setup into a standalone Python script.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    if fn is not None and bounds_ok and not compile_error and not has_constraint_error:
        script = generate_script(
            code,
            name=problem_name,
            n_var=int(n_var),
            n_obj=int(n_obj),
            bounds=bounds_ok,
            algorithm=algorithm,
            budget=budget,
            constraint_code=constraint_code,
            n_constraints=n_constraints,
        )
        st.download_button(
            "Download as Python script",
            data=script,
            file_name=f"{problem_name.replace(' ', '_').lower()}.py",
            mime="text/x-python",
            help="Download a standalone .py file you can run with `python <file>.py`.",
        )
        if advanced_mode:
            with st.expander("Preview generated script"):
                st.code(script, language="python")
    else:
        if not code.strip():
            st.info("Write your objective function above to enable export.")
        elif compile_error or has_constraint_error:
            st.warning("Fix the errors above to enable export.")
        elif not bounds_ok:
            st.warning("Fix the bounds above to enable export.")
        else:
            st.info("Complete the fields above to enable export.")


def _run_and_show_preview(
    st: Any,
    px: Any,
    *,
    fn: Any,
    objective_code: str,
    constraint_code: str,
    problem_name: str,
    n_var: int,
    n_obj: int,
    bounds_ok: list[tuple[float, float]],
    algorithm: str,
    budget: int,
    pop_size: int,
    seed: int,
    constraints: Any = None,
    n_constraints: int = 0,
) -> None:
    """Execute the preview optimization and render charts."""
    try:
        with st.spinner("Running optimization..."):
            preview = run_preview_optimization(
                fn,
                n_var=n_var,
                n_obj=n_obj,
                bounds=bounds_ok,
                algorithm=algorithm,
                budget=budget,
                pop_size=pop_size,
                seed=seed,
                constraints=constraints,
                n_constraints=n_constraints,
                objective_code=objective_code,
                constraint_code=constraint_code,
            )
        F = preview["F"]
        st.success(f"Found {len(F)} solutions in {preview['elapsed_ms']:.0f} ms")
        _render_preview_guidance(st, n_obj)
        _render_preview_plot(st, px, F, n_obj, problem_name)
        _render_summary_table(st, F, n_obj)
    except TimeoutError as exc:
        st.error(str(exc))
        st.info("Try reducing max evaluations, simplifying your objective function, or removing expensive loops.")
    except Exception as exc:
        st.error(f"Optimization failed: {exc}")
        st.info("Common causes: your function returns the wrong number of objectives, or uses variables outside the bounds.")


__all__ = ["render_problem_builder"]
