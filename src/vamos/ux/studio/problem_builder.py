"""VAMOS Studio -- Problem Builder page (Streamlit UI layer).

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
        default_bounds = template.get("bounds", "0.0, 1.0")
        bounds_text = st.text_area(
            "Variable bounds",
            value=default_bounds,
            height=68,
            help="One line applies to ALL variables.  Or one 'lower, upper' per variable.",
        )
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

    with right:
        st.subheader("Optimization settings")
        col_algo, col_seed = st.columns(2)
        with col_algo:
            algo_labels = {
                "nsgaii": "NSGA-II (general purpose, recommended)",
                "moead": "MOEA/D (good for decomposition-style searches)",
                "spea2": "SPEA2 (strong diversity maintenance)",
                "smsemoa": "SMS-EMOA (hypervolume-driven selection)",
                "nsgaiii": "NSGA-III (many-objective scenarios)",
                "ibea": "IBEA (indicator-based)",
                "agemoea": "AGE-MOEA (geometry-aware)",
                "rvea": "RVEA (reference-vector guided)",
            }
            algorithm = st.selectbox(
                "Algorithm",
                list(algo_labels.keys()),
                index=0,
                format_func=lambda key: algo_labels.get(key, key),
                help="Pick a MOEA for preview runs. Start with NSGA-II unless you have a specific reason to choose another.",
            )
        with col_seed:
            seed = st.number_input("Seed", min_value=0, value=42, step=1, help="Random seed for reproducibility.")
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

        has_constraint_error = constraint_error is not None
        # Live preview
        if run_clicked and fn is not None and bounds_ok and not compile_error and not has_constraint_error:
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
                algorithm=str(algorithm),
                budget=int(budget),
                pop_size=int(pop_size),
                seed=int(seed),
                constraints=constraint_fn,
                n_constraints=n_constraints,
            )

    # ---- export ----
    st.divider()
    st.subheader("Export")
    if fn is not None and bounds_ok and not compile_error and not has_constraint_error:
        script = generate_script(
            code,
            name=problem_name,
            n_var=int(n_var),
            n_obj=int(n_obj),
            bounds=bounds_ok,
            algorithm=str(algorithm),
            budget=int(budget),
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
        _render_preview_plot(st, px, F, n_obj, problem_name)
        _render_summary_table(st, F, n_obj)
    except TimeoutError as exc:
        st.error(str(exc))
        st.info("Try reducing max evaluations, simplifying your objective function, or removing expensive loops.")
    except Exception as exc:
        st.error(f"Optimization failed: {exc}")
        st.info("Common causes: your function returns the wrong number of objectives, or uses variables outside the bounds.")


__all__ = ["render_problem_builder"]
