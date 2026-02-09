"""VAMOS Studio -- Problem Builder page (Streamlit UI layer).

Delegates heavy lifting to ``problem_builder_backend``.
"""

from __future__ import annotations

from typing import Any

from vamos.ux.studio.problem_builder_backend import (
    _DEFAULT_TEMPLATE,
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


def _render_preview_plot(
    st: Any,
    px: Any,
    F: Any,
    n_obj: int,
    problem_name: str,
) -> None:
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


# ------------------------------------------------------------------
# Main Streamlit entry
# ------------------------------------------------------------------


def render_problem_builder(st: Any, px: Any) -> None:
    """Render the Problem Builder tab inside VAMOS Studio."""
    st.header("Problem Builder")
    st.caption(
        "Define your objectives, adjust parameters, and see the Pareto front update live.  When happy, export a ready-to-run Python script."
    )

    # ---- template selector ----
    examples = example_objectives()
    col_tmpl, _ = st.columns([2, 3])
    with col_tmpl:
        template_name = st.selectbox(
            "Start from a template",
            list(examples.keys()),
            index=list(examples.keys()).index(_DEFAULT_TEMPLATE),
            help="Pick a starting point. You can edit everything below.",
        )
    template = examples[template_name]

    # ---- form columns ----
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Problem definition")
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
        bounds_text = st.text_area(
            "Variable bounds",
            value="0.0, 1.0",
            height=68,
            help="One line applies to ALL variables.  Or one 'lower, upper' per variable.",
        )
        st.markdown("**Objective function** (receives `x`, return a list of objectives)")
        code = st.text_area(
            "Objective code",
            value=template["code"],
            height=220,
            help=(
                "Write Python code that uses `x` (1-D array of length n_var) "
                "and returns a list of n_obj values.  `math` and `np` are available."
            ),
            label_visibility="collapsed",
        )

    with right:
        st.subheader("Optimization settings")
        col_algo, col_seed = st.columns(2)
        with col_algo:
            algorithm = st.selectbox(
                "Algorithm",
                ["nsgaii", "moead", "spea2", "smsemoa", "nsgaiii", "ibea", "agemoea", "rvea"],
                index=0,
                help="Which MOEA to use for the preview run.",
            )
        with col_seed:
            seed = st.number_input(
                "Seed",
                min_value=0,
                value=42,
                step=1,
                help="Random seed for reproducibility.",
            )
        col_budget, col_pop = st.columns(2)
        with col_budget:
            budget = st.number_input(
                "Max evaluations",
                min_value=200,
                max_value=50000,
                value=2000,
                step=500,
                help="Higher = better results, slower preview.",
            )
        with col_pop:
            pop_size = st.number_input(
                "Population size",
                min_value=10,
                max_value=500,
                value=50,
                step=10,
                help="Number of solutions per generation.",
            )
        run_clicked = st.button("Run preview", type="primary", use_container_width=True)

        # Validation
        bounds_result = parse_bounds_text(bounds_text, int(n_var))
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

        # Live preview
        if run_clicked and fn is not None and bounds_ok and not compile_error:
            _run_and_show_preview(
                st,
                px,
                fn=fn,
                problem_name=problem_name,
                n_var=int(n_var),
                n_obj=int(n_obj),
                bounds_ok=bounds_ok,
                algorithm=str(algorithm),
                budget=int(budget),
                pop_size=int(pop_size),
                seed=int(seed),
            )

    # ---- export ----
    st.divider()
    st.subheader("Export")
    if fn is not None and bounds_ok and not compile_error:
        script = generate_script(
            code,
            name=problem_name,
            n_var=int(n_var),
            n_obj=int(n_obj),
            bounds=bounds_ok,
            algorithm=str(algorithm),
            budget=int(budget),
        )
        st.download_button(
            "Download as Python script",
            data=script,
            file_name=f"{problem_name.replace(' ', '_').lower()}.py",
            mime="text/x-python",
            help="Download a standalone .py file you can run with `python <file>.py`.",
        )
        with st.expander("Preview generated script"):
            st.code(script, language="python")
    else:
        st.info("Fix any errors above to enable export.")


def _run_and_show_preview(
    st: Any,
    px: Any,
    *,
    fn: Any,
    problem_name: str,
    n_var: int,
    n_obj: int,
    bounds_ok: list[tuple[float, float]],
    algorithm: str,
    budget: int,
    pop_size: int,
    seed: int,
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
            )
        F = preview["F"]
        st.success(f"Found {len(F)} solutions in {preview['elapsed_ms']:.0f} ms")
        _render_preview_plot(st, px, F, n_obj, problem_name)
        _render_summary_table(st, F, n_obj)
    except Exception as exc:
        st.error(f"Optimization failed: {exc}")
        st.info("Common causes: your function returns the wrong number of objectives, or uses variables outside the bounds.")


__all__ = ["render_problem_builder"]
