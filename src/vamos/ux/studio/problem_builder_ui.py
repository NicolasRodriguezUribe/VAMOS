"""Reusable UI helpers for the Studio problem builder."""

from __future__ import annotations

from typing import Any

from vamos.ux.studio.problem_builder_backend import compile_constraint_function

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


def render_builder_intro(st: Any) -> None:
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


def render_preview_plot(st: Any, px: Any, F: Any, n_obj: int, problem_name: str) -> None:
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
        return

    if n_obj == 3:
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
        return

    import pandas as pd

    cols = [f"f{i}" for i in range(n_obj)]
    df_pc = pd.DataFrame(F, columns=cols)
    fig_pc = px.parallel_coordinates(df_pc, dimensions=cols, title="Pareto Front (parallel coordinates)")
    fig_pc.update_layout(height=420)
    st.plotly_chart(fig_pc, use_container_width=True)


def render_summary_table(st: Any, F: Any, n_obj: int) -> None:
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


def coerce_bool_widget_value(value: Any, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def render_template_snapshot(st: Any, template: dict[str, str]) -> None:
    constraints = int(template.get("n_constraints", "0"))
    col_var, col_obj, col_con = st.columns(3)
    with col_var:
        st.metric("Variables", template["n_var"])
    with col_obj:
        st.metric("Objectives", template["n_obj"])
    with col_con:
        st.metric("Constraints", str(constraints))


def render_preview_guidance(st: Any, n_obj: int) -> None:
    if n_obj == 2:
        st.info(
            "How to read this chart: each point is one solution. Points farther down and left are usually better, "
            "but no single point wins on every goal."
        )
        return
    st.info(
        "How to read this preview: Studio is showing a compact view of several goals at once. "
        "Use the summary table to compare the spread of objective values."
    )


def render_template_chips(st: Any, template: dict[str, str], *, advanced_mode: bool) -> None:
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


def render_constraint_section(st: Any, template: dict[str, str]) -> tuple[str, int, Any, str | None]:
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


__all__ = [
    "_ALGO_HELP",
    "_ALGO_LABELS",
    "coerce_bool_widget_value",
    "render_builder_intro",
    "render_constraint_section",
    "render_preview_guidance",
    "render_preview_plot",
    "render_summary_table",
    "render_template_chips",
    "render_template_snapshot",
]
