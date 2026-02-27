"""VAMOS Studio -- Explore Results tab.

Progressive disclosure: essential controls (directory, problem, algorithms)
stay at the top of the sidebar.  Ranking preferences (weights, reference
points, MCDM method) are tucked inside an expander for intermediate and
advanced users.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.ux.studio.services import (
    build_demo_study_data,
    build_decision_views,
    discover_study_directories,
    load_studio_data,
    run_focused_optimization,
    run_with_history,
)

if TYPE_CHECKING:
    from vamos.ux.studio.data import FrontRecord


# ------------------------------------------------------------------
# MCDM friendly labels
# ------------------------------------------------------------------

_MCDM_LABELS = {
    "weighted_sum": "Simple weighted score",
    "tchebycheff": "Balanced compromise",
    "knee": "Natural trade-off point",
    "topsis": "Closest to the ideal target",
}


def _select_primary_front(
    fronts: list[FrontRecord],
    problem: str,
    primary_algo: str | None,
) -> FrontRecord | None:
    if primary_algo is None:
        return None
    return next(
        (f for f in fronts if f.problem_name == problem and f.algorithm_name == primary_algo),
        None,
    )


def _format_study_dir_label(path: Path, base_dir: Path) -> str:
    try:
        rel_path = path.resolve().relative_to(base_dir.resolve())
        return str(rel_path)
    except ValueError:
        return str(path)


def _render_explore_intro(st: Any) -> None:
    """Render the Explore Results lead-in copy."""
    st.markdown(
        '<div class="studio-section-intro">'
        '<span class="studio-kicker">Explore</span>'
        "<h3>Read the trade-offs before you pick a favorite</h3>"
        "<p>Compare fronts visually, then tell Studio which goals matter more so it can surface the solutions that best match your priorities.</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_dataset_overview(
    st: Any,
    *,
    using_demo: bool,
    problem_count: int,
    algorithm_count: int,
    front_count: int,
) -> None:
    """Show a compact strip describing the loaded dataset."""
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


# ------------------------------------------------------------------
# Sub-sections
# ------------------------------------------------------------------


def _render_pareto_scatter(
    st: Any,
    px: Any,
    comparison_fronts: list[FrontRecord],
    primary_front: FrontRecord,
    obj_idx: list[int],
    order: Any,
    method: str,
    problem: str,
) -> None:
    """Render the 2-D Pareto front scatter chart."""
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
    fig = px.scatter(
        df,
        x=f"f{obj_idx[0]}",
        y=f"f{obj_idx[1]}",
        color="algorithm",
        title=f"{problem}: trade-off view",
    )

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


def _render_top_solutions(
    st: Any,
    primary_front: FrontRecord,
    primary_algo: str,
    order: Any,
    view: Any,
    export_dir: Path,
) -> None:
    """Render the top-k solution table + export."""
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


def _render_advanced_sections(
    st: Any,
    px: Any,
    *,
    view: Any,
    scores: Any,
    reference_point: Any,
    problem: str,
    primary_algo: str,
    primary_front: FrontRecord,
    obj_idx: list[int],
    is_demo: bool,
    focus_budget_default: int = 500,
) -> None:
    """Render parallel-coordinates, focused optimization and search dynamics."""
    focus_budget = int(focus_budget_default)

    # Parallel coordinates
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

    # Focused optimization
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

    # Search dynamics
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
    primary_front: FrontRecord,
    problem: str,
    primary_algo: str,
    obj_idx: list[int],
    budget: int,
) -> None:
    """Run and animate the optimization evolution."""
    import pandas as pd

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


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------


def render_explore_tab(st: Any, px: Any, study_dir: Path) -> None:
    """Render the full Explore Results tab."""
    _render_explore_intro(st)
    workspace_dir = Path.cwd()
    default_study_dir = study_dir.expanduser()
    recent_dirs = discover_study_directories(workspace_dir)
    selected_study_dir = default_study_dir

    if recent_dirs:
        default_index = 0
        if default_study_dir.exists():
            default_resolved = default_study_dir.resolve()
            for idx, candidate in enumerate(recent_dirs):
                if candidate.resolve() == default_resolved:
                    default_index = idx
                    break
        st.sidebar.caption("Open one of the detected result folders, or fall back to the demo.")
        selected_study_dir = Path(
            st.sidebar.selectbox(
                "Saved runs",
                recent_dirs,
                index=default_index,
                format_func=lambda path: _format_study_dir_label(Path(path), workspace_dir),
                help="Studio looks for result folders under the current workspace.",
            )
        )
    else:
        st.sidebar.info("No saved runs were detected yet. Studio will show a built-in demo unless you choose another folder.")

    with st.sidebar.expander("Use another folder", expanded=False):
        custom_study_input = st.text_input(
            "Results folder",
            value="",
            placeholder=str(default_study_dir),
            help="Optional: point Studio to a different results folder.",
        ).strip()

    effective_study_dir = Path(custom_study_input).expanduser() if custom_study_input else selected_study_dir
    export_dir = effective_study_dir if effective_study_dir.exists() else workspace_dir

    runs: list[Any] = []
    fronts: list[FrontRecord] = []
    using_demo = False
    if effective_study_dir.exists():
        runs, fronts = load_studio_data(effective_study_dir)
    if not fronts:
        using_demo = True
        runs, fronts = build_demo_study_data()
        st.sidebar.info("Showing built-in demo data.")
        if effective_study_dir.exists():
            st.info(
                f"No results were found in `{effective_study_dir}`. Showing a built-in demo so you can explore the interface right away."
            )
        else:
            st.info(
                f"Results folder `{effective_study_dir}` was not found. Showing a built-in demo so you can explore the interface right away."
            )
    else:
        st.sidebar.success(f"Loaded {len(runs)} runs across {len(fronts)} fronts.")

    _render_dataset_overview(
        st,
        using_demo=using_demo,
        problem_count=len({f.problem_name for f in fronts}),
        algorithm_count=len({f.algorithm_name for f in fronts}),
        front_count=len(fronts),
    )
    st.markdown(
        '<div class="studio-data-strip">'
        "<p>Use the sidebar to switch datasets and adjust how Studio chooses a favorite solution. The chart below stays focused on just two goals so the trade-off is easy to read.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ---- Essential sidebar controls (visible to all) ----
    problems = sorted({f.problem_name for f in fronts})
    problem = st.sidebar.selectbox(
        "Problem or scenario",
        problems,
        help="Choose which optimization problem to inspect.",
    )
    algos = sorted({f.algorithm_name for f in fronts if f.problem_name == problem})
    selected_algos = st.sidebar.multiselect(
        "Algorithms to compare",
        algos,
        default=algos[:1],
        help="Pick one or more algorithms to compare side by side.",
    )
    primary_algo = (
        st.sidebar.selectbox(
            "Which algorithm should Studio rank?",
            selected_algos,
            help="Studio will rank solutions from this algorithm in the table below.",
        )
        if selected_algos
        else None
    )
    if not selected_algos:
        st.info("Select at least one algorithm in the sidebar to visualize.")
        return

    comparison_fronts = [f for f in fronts if f.problem_name == problem and f.algorithm_name in selected_algos]
    primary_front = _select_primary_front(fronts, problem, primary_algo)
    if primary_front is None:
        st.error("Primary front not found. Try selecting a different algorithm.")
        return
    if primary_algo is None:
        st.error("Primary algorithm not selected.")
        return

    # ---- Ranking preferences (advanced — in expander) ----
    default_weights = np.ones(primary_front.points_F.shape[1]) / primary_front.points_F.shape[1]
    weights = default_weights.copy()
    reference_point = None
    method = "weighted_sum"

    with st.sidebar.expander("How should Studio pick a favorite?", expanded=False):
        st.caption(
            "Tell Studio what matters most. Higher weights mean a goal matters more. "
            "You can also set ideal target values if you have them."
        )
        weight_inputs = []
        for i in range(primary_front.points_F.shape[1]):
            weight_inputs.append(
                st.slider(
                    f"Importance of goal {i + 1}",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(default_weights[i]),
                    step=0.05,
                    help=f"Higher means goal {i + 1} matters more in the ranking.",
                )
            )
        weights = np.array(weight_inputs)
        if weights.sum() == 0:
            weights = default_weights

        ref_input = st.text_input(
            "Ideal target values (optional)",
            "",
            help="Optional comma-separated target values, one for each goal.",
        )
        if ref_input.strip():
            try:
                reference_point = np.array([float(v.strip()) for v in ref_input.split(",")])
            except Exception:
                st.error("Invalid format. Enter comma-separated numbers.")

        method = str(
            st.selectbox(
                "Ranking style",
                list(_MCDM_LABELS.keys()),
                format_func=lambda key: _MCDM_LABELS.get(key, key),
                help="Simple weighted score is easiest to start with.",
            )
        )

    views = build_decision_views([primary_front], weights, reference_point, method)
    view = views[0]
    scores = view.mcdm_scores.get(method, np.zeros(view.front.points_F.shape[0]))
    order = np.argsort(scores)

    # ---- Main content ----
    st.subheader("Trade-off comparison")
    st.caption(
        "Each point is one solution. The star marks the solution Studio currently favors based on your choices."
    )
    if using_demo:
        st.info("You are viewing demo data. Open a saved run from the sidebar when you have one.")
    obj_idx = st.multiselect(
        "Goals to plot (pick 2)",
        list(range(primary_front.points_F.shape[1])),
        default=[0, 1],
        help="Pick exactly two goals to draw the 2-D chart.",
    )
    if len(obj_idx) != 2:
        st.info("Pick exactly two goals to draw the 2-D chart.")
    else:
        _render_pareto_scatter(st, px, comparison_fronts, primary_front, obj_idx, order, method, problem)

    # Top solutions table
    _render_top_solutions(st, primary_front, primary_algo, order, view, export_dir)

    # Advanced sections
    _render_advanced_sections(
        st,
        px,
        view=view,
        scores=scores,
        reference_point=reference_point,
        problem=problem,
        primary_algo=primary_algo,
        primary_front=primary_front,
        obj_idx=obj_idx,
        is_demo=using_demo,
    )


__all__ = ["render_explore_tab"]
