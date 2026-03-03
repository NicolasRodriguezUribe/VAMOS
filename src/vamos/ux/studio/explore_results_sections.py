"""UI sections used by the Explore Results Studio tab."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vamos.ux.studio.data import FrontRecord

_MCDM_LABELS = {
    "weighted_sum": "Simple weighted score",
    "tchebycheff": "Balanced compromise",
    "knee": "Natural trade-off point",
    "topsis": "Closest to the ideal target",
}


def render_explore_intro(st: Any) -> None:
    st.markdown(
        '<div class="studio-section-intro">'
        '<span class="studio-kicker">Explore</span>'
        "<h3>Read the trade-offs before you pick a favorite</h3>"
        "<p>Compare fronts visually, then tell Studio which goals matter more so it can surface the solutions that best match your priorities.</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_dataset_overview(
    st: Any,
    *,
    using_demo: bool,
    problem_count: int,
    algorithm_count: int,
    front_count: int,
) -> None:
    source_label = "Built-in demo" if using_demo else "Saved study"
    st.markdown(
        '<div class="studio-chip-row">'
        f'<span class="studio-chip">{source_label}</span>'
        f'<span class="studio-chip">{problem_count} problem(s)</span>'
        f'<span class="studio-chip">{algorithm_count} algorithm(s)</span>'
        f'<span class="studio-chip">{front_count} front(s)</span>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_pareto_scatter(
    st: Any,
    px: Any,
    comparison_fronts: list["FrontRecord"],
    primary_front: "FrontRecord",
    obj_idx: list[int],
    order: Any,
    method: str,
    problem: str,
) -> None:
    if len(obj_idx) != 2:
        return
    import pandas as pd

    plot_data = []
    for front in comparison_fronts:
        for i in range(front.points_F.shape[0]):
            plot_data.append(
                {
                    f"f{obj_idx[0]}": float(front.points_F[i, obj_idx[0]]),
                    f"f{obj_idx[1]}": float(front.points_F[i, obj_idx[1]]),
                    "algorithm": front.algorithm_name,
                }
            )

    df = pd.DataFrame(plot_data)
    fig = px.scatter(df, x=f"f{obj_idx[0]}", y=f"f{obj_idx[1]}", color="algorithm", title=f"{problem}: trade-off view")
    best_idx = int(order[0])
    best_point = primary_front.points_F[best_idx]
    fig.add_scatter(
        x=[best_point[obj_idx[0]]],
        y=[best_point[obj_idx[1]]],
        mode="markers",
        marker=dict(size=12, color="red", symbol="star", line=dict(width=2, color="black")),
        name=f"Favorite ({_MCDM_LABELS.get(method, method)})",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_top_solutions(
    st: Any,
    primary_front: "FrontRecord",
    primary_algo: str,
    order: Any,
    view: Any,
    export_dir: Path,
) -> None:
    import pandas as pd

    st.subheader(f"Best matches ({primary_algo})")
    top_k = st.slider(
        "How many matches should Studio show?",
        min_value=1,
        max_value=min(20, len(order)),
        value=min(5, len(order)),
        help="Show more or fewer high-ranked solutions in the table below.",
    )
    top_indices = order[:top_k]
    st.dataframe(
        pd.DataFrame(
            primary_front.points_F[top_indices],
            columns=[f"f{i}" for i in range(primary_front.points_F.shape[1])],
        ),
        use_container_width=True,
    )
    if st.button("Export selected solutions as JSON", help="Save the selected solutions to a JSON file."):
        from vamos.ux.studio.export import export_solutions_to_json

        path = export_solutions_to_json(view, top_indices.tolist(), export_dir / "studio_export.json")
        st.success(f"Exported to {path}")


def render_advanced_sections(
    st: Any,
    px: Any,
    *,
    view: Any,
    scores: Any,
    reference_point: Any,
    problem: str,
    primary_algo: str,
    primary_front: "FrontRecord",
    obj_idx: list[int],
    is_demo: bool,
    focus_budget_default: int = 500,
) -> None:
    focus_budget = int(focus_budget_default)
    with st.expander("Compare all goals (advanced)", expanded=False):
        try:
            fig_pc = px.parallel_coordinates(
                view.normalized_F,
                color=scores,
                labels={i: f"f{i}" for i in range(view.normalized_F.shape[1])},
            )
            st.plotly_chart(fig_pc, use_container_width=True)
        except Exception as exc:
            st.warning(f"Parallel coordinates unavailable: {exc}")

    with st.expander("Focused re-run", expanded=False):
        if is_demo:
            st.info("Focused re-runs are available for saved studies, not for the built-in demo.")
        else:
            st.caption("Re-run the algorithm around your target values to focus the search.")
            focus_budget = int(
                st.number_input(
                    "Budget (evaluations)",
                    min_value=100,
                    max_value=5000,
                    value=focus_budget_default,
                    step=100,
                    help="Number of function evaluations for the focused run.",
                )
            )
            if st.button("Run focused re-run"):
                if reference_point is None:
                    st.error("Set target values in the sidebar first.")
                else:
                    from vamos.ux.studio.services import run_focused_optimization

                    with st.spinner("Running focused optimization..."):
                        F_new, _ = run_focused_optimization(problem, reference_point, primary_algo, int(focus_budget))
                    st.success(f"Focused run produced {len(F_new)} points.")
                    if len(obj_idx) == 2:
                        fig2 = px.scatter(
                            x=F_new[:, obj_idx[0]],
                            y=F_new[:, obj_idx[1]],
                            labels={"x": f"f{obj_idx[0]}", "y": f"f{obj_idx[1]}"},
                        )
                        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Replay optimization history", expanded=False):
        if is_demo:
            st.info("Replay is available only for saved runs that include configuration metadata.")
        elif primary_front.extra.get("config"):
            st.caption("Re-run and animate how the population moves generation by generation.")
            if len(obj_idx) != 2:
                st.info("Pick exactly two goals above before replaying the 2-D animation.")
            elif st.button("Replay optimization"):
                _animate_evolution(st, px, primary_front, problem, primary_algo, obj_idx, int(focus_budget))
        else:
            st.info("Config not available for this run (cannot re-run accurately).")


def _animate_evolution(
    st: Any,
    px: Any,
    primary_front: "FrontRecord",
    problem: str,
    primary_algo: str,
    obj_idx: list[int],
    budget: int,
) -> None:
    import pandas as pd

    from vamos.ux.studio.services import run_with_history

    config = primary_front.extra["config"]
    with st.spinner("Re-running optimization to capture history..."):
        _, history = run_with_history(problem, config, budget)
    if not history:
        st.warning("No history captured. Ensure algorithm supports callbacks.")
        return

    st.success(f"Captured {len(history)} generations.")
    frames = []
    for gen, F_gen in enumerate(history):
        if len(F_gen) > 200:
            F_gen = F_gen[:200]
        for i in range(len(F_gen)):
            frames.append(
                {
                    "Generation": gen,
                    f"f{obj_idx[0]}": float(F_gen[i, obj_idx[0]]),
                    f"f{obj_idx[1]}": float(F_gen[i, obj_idx[1]]),
                }
            )
    df_anim = pd.DataFrame(frames)
    fig_anim = px.scatter(
        df_anim,
        x=f"f{obj_idx[0]}",
        y=f"f{obj_idx[1]}",
        animation_frame="Generation",
        range_x=[df_anim[f"f{obj_idx[0]}"].min(), df_anim[f"f{obj_idx[0]}"].max()],
        range_y=[df_anim[f"f{obj_idx[1]}"].min(), df_anim[f"f{obj_idx[1]}"].max()],
        title=f"Evolution of {primary_algo} on {problem}",
    )
    st.plotly_chart(fig_anim, use_container_width=True)


__all__ = [
    "_MCDM_LABELS",
    "render_advanced_sections",
    "render_dataset_overview",
    "render_explore_intro",
    "render_pareto_scatter",
    "render_top_solutions",
]
