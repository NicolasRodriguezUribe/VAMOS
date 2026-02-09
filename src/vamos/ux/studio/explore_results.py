"""VAMOS Studio -- Explore Results tab.

Extracted from ``app.py`` to keep each module under the LOC budget.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.ux.studio.services import (
    build_decision_views,
    load_studio_data,
    run_focused_optimization,
    run_with_history,
)

if TYPE_CHECKING:
    from vamos.ux.studio.data import FrontRecord


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


# ------------------------------------------------------------------
# Sidebar: load data + preferences
# ------------------------------------------------------------------


def _render_sidebar(st: Any, study_dir: Path) -> tuple[Any, list[Any], str | None, list[str], np.ndarray, Any, str, Any]:
    """Render sidebar widgets and return parsed state.

    Returns (fronts, comparison_fronts, primary_algo, obj_idx_default,
    weights, reference_point, method, primary_front) or raises early-return
    sentinel via ``_EarlyReturn``.
    """
    study_dir = Path(
        st.sidebar.text_input(
            "Study directory",
            str(study_dir),
            help="Path to the folder containing optimization results (e.g. results/).",
        )
    ).expanduser()

    if not study_dir.exists():
        return None, study_dir  # type: ignore[return-value]

    runs, fronts = load_studio_data(study_dir)
    st.sidebar.success(f"Loaded {len(runs)} runs across {len(fronts)} fronts.")
    return fronts, study_dir  # type: ignore[return-value]


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
        title=f"{problem} Pareto Fronts",
    )

    best_idx = int(order[0])
    best_point = primary_front.points_F[best_idx]
    fig.add_scatter(
        x=[best_point[obj_idx[0]]],
        y=[best_point[obj_idx[1]]],
        mode="markers",
        marker=dict(size=12, color="red", symbol="star", line=dict(width=2, color="black")),
        name=f"Best ({method})",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_top_solutions(
    st: Any,
    primary_front: FrontRecord,
    primary_algo: str,
    order: Any,
    view: Any,
    study_dir: Path,
) -> None:
    """Render the top-k solution table + export."""
    import pandas as pd

    st.subheader(f"Top solutions ({primary_algo})")
    top_k = st.slider(
        "Show top-k",
        min_value=1,
        max_value=min(20, len(order)),
        value=min(5, len(order)),
        help="How many top-ranked solutions to display in the table below.",
    )
    top_indices = order[:top_k]
    st.dataframe(
        pd.DataFrame(
            primary_front.points_F[top_indices],
            columns=[f"f{i}" for i in range(primary_front.points_F.shape[1])],
        ),
        use_container_width=True,
    )
    if st.button("Export top-k as JSON", help="Save the selected solutions to a JSON file."):
        from vamos.ux.studio.export import export_solutions_to_json

        path = export_solutions_to_json(view, top_indices.tolist(), study_dir / "studio_export.json")
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
    focus_budget_default: int = 500,
) -> None:
    """Render parallel-coordinates, focused optimization and search dynamics."""
    # Parallel coordinates
    with st.expander("Parallel Coordinates", expanded=False):
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
    with st.expander("Focused Optimization", expanded=False):
        st.caption("Re-run the algorithm with a reference point to focus the search.")
        focus_budget = st.number_input(
            "Budget (evaluations)",
            min_value=100,
            max_value=5000,
            value=focus_budget_default,
            step=100,
            help="Number of function evaluations for the focused run.",
        )
        if st.button("Run focused optimization"):
            if reference_point is None:
                st.error("Set a reference point in the sidebar first.")
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
    with st.expander("Search Dynamics (Re-run)", expanded=False):
        if primary_front.extra.get("config"):
            st.caption("Re-run and animate how the population evolves generation by generation.")
            if st.button("Animate Optimization Evolution"):
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
    study_dir = Path(
        st.sidebar.text_input(
            "Study directory",
            str(study_dir),
            help="Path to the folder containing optimization results (e.g. results/).",
        )
    ).expanduser()

    if not study_dir.exists():
        st.warning(f"Study directory `{study_dir}` not found. Run an optimization first (`vamos quickstart`) or check the path.")
        return

    runs, fronts = load_studio_data(study_dir)
    st.sidebar.success(f"Loaded {len(runs)} runs across {len(fronts)} fronts.")

    if not fronts:
        st.info(
            "No results found in this directory. Run an optimization first:\n\n"
            "```bash\nvamos quickstart\n```\n\n"
            "Then come back and point the study directory to `results/quickstart`."
        )
        return

    problems = sorted({f.problem_name for f in fronts})
    problem = st.sidebar.selectbox(
        "Problem",
        problems,
        help="Select which optimization problem to inspect.",
    )
    algos = sorted({f.algorithm_name for f in fronts if f.problem_name == problem})
    selected_algos = st.sidebar.multiselect(
        "Algorithms to compare",
        algos,
        default=algos[:1],
        help="Pick one or more algorithms to visualize side by side.",
    )
    primary_algo = (
        st.sidebar.selectbox(
            "Primary algorithm (for MCDM)",
            selected_algos,
            help="The algorithm whose solutions are ranked by the MCDM method below.",
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

    # Preferences
    default_weights = np.ones(primary_front.points_F.shape[1]) / primary_front.points_F.shape[1]
    weight_inputs = []
    st.sidebar.subheader("Preferences")
    for i in range(primary_front.points_F.shape[1]):
        weight_inputs.append(
            st.sidebar.slider(
                f"Weight f{i}",
                min_value=0.0,
                max_value=1.0,
                value=float(default_weights[i]),
                step=0.05,
                help=f"Relative importance of objective f{i} (higher = more important).",
            )
        )
    weights = np.array(weight_inputs)
    if weights.sum() == 0:
        weights = default_weights

    ref_input = st.sidebar.text_input(
        "Reference point (comma-separated)",
        "",
        help="An aspiration point for Tchebycheff / reference-point methods.",
    )
    reference_point = None
    if ref_input.strip():
        try:
            reference_point = np.array([float(v.strip()) for v in ref_input.split(",")])
        except Exception:
            st.sidebar.error("Invalid format. Enter comma-separated numbers.")

    method = st.sidebar.selectbox(
        "MCDM method",
        ["tchebycheff", "weighted_sum", "knee", "topsis"],
        help="Multi-Criteria Decision Making method to rank solutions.",
    )

    views = build_decision_views([primary_front], weights, reference_point, method)
    view = views[0]
    scores = view.mcdm_scores.get(method, np.zeros(view.front.points_F.shape[0]))
    order = np.argsort(scores)

    # Pareto front scatter
    st.subheader("Pareto Front Comparison")
    obj_idx = st.multiselect(
        "Objectives to plot (choose 2)",
        list(range(primary_front.points_F.shape[1])),
        default=[0, 1],
        help="Select exactly two objectives for a 2-D scatter plot.",
    )
    _render_pareto_scatter(st, px, comparison_fronts, primary_front, obj_idx, order, method, problem)

    # Top solutions table
    _render_top_solutions(st, primary_front, primary_algo, order, view, study_dir)

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
    )


__all__ = ["render_explore_tab"]
