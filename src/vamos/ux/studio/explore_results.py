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

from vamos.ux.studio.explore_results_sections import (
    _MCDM_LABELS,
    render_advanced_sections,
    render_dataset_overview,
    render_explore_intro,
    render_pareto_scatter,
    render_top_solutions,
)
from vamos.ux.studio.services import (
    build_demo_study_data,
    build_decision_views,
    discover_study_directories,
    load_studio_data,
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


def _format_study_dir_label(path: Path, base_dir: Path) -> str:
    try:
        rel_path = path.resolve().relative_to(base_dir.resolve())
        return str(rel_path)
    except ValueError:
        return str(path)


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------


def render_explore_tab(st: Any, px: Any, study_dir: Path) -> None:
    """Render the full Explore Results tab."""
    render_explore_intro(st)
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

    render_dataset_overview(
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
        render_pareto_scatter(st, px, comparison_fronts, primary_front, obj_idx, order, method, problem)

    # Top solutions table
    render_top_solutions(st, primary_front, primary_algo, order, view, export_dir)

    # Advanced sections
    render_advanced_sections(
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
