from __future__ import annotations

import argparse
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


def _import_streamlit() -> Any:
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("VAMOS Studio requires the 'studio' extra: pip install -e \".[studio]\"") from exc
    return st


def _import_plotly() -> Any:
    try:
        import plotly.express as px
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Plotly is required for interactive plots. Install with the 'studio' extras.") from exc
    return px


def _select_primary_front(
    fronts: list[FrontRecord],
    problem: str,
    primary_algo: str | None,
) -> FrontRecord | None:
    if primary_algo is None:
        return None
    return next((f for f in fronts if f.problem_name == problem and f.algorithm_name == primary_algo), None)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Launch VAMOS Studio (Streamlit).")
    parser.add_argument("--study-dir", help="Path to a StudyRunner output directory.", default="results")
    args, _ = parser.parse_known_args(argv)

    st = _import_streamlit()
    px = _import_plotly()

    st.set_page_config(page_title="VAMOS Studio", layout="wide")
    st.title("VAMOS Studio")
    study_dir = Path(st.sidebar.text_input("Study directory", args.study_dir)).expanduser()
    if not study_dir.exists():
        st.warning(f"Study directory {study_dir} not found.")
        return

    runs, fronts = load_studio_data(study_dir)
    st.sidebar.success(f"Loaded {len(runs)} runs across {len(fronts)} fronts.")

    problems = sorted({f.problem_name for f in fronts})
    problem = st.sidebar.selectbox("Problem", problems)
    algos = sorted({f.algorithm_name for f in fronts if f.problem_name == problem})

    selected_algos = st.sidebar.multiselect("Algorithms to Compare", algos, default=algos[:1])
    primary_algo = st.sidebar.selectbox("Primary Algorithm (for MCDM)", selected_algos) if selected_algos else None

    if not selected_algos:
        st.info("Select at least one algorithm to visualize.")
        return

    comparison_fronts = [f for f in fronts if f.problem_name == problem and f.algorithm_name in selected_algos]
    primary_front = _select_primary_front(fronts, problem, primary_algo)

    if primary_front is None:
        st.error("Primary front not found.")
        return
    if primary_algo is None:
        st.error("Primary algorithm not selected.")
        return

    default_weights = np.ones(primary_front.points_F.shape[1]) / primary_front.points_F.shape[1]
    weight_inputs = []
    st.sidebar.subheader("Preferences (Primary Algo)")
    for i in range(primary_front.points_F.shape[1]):
        weight_inputs.append(st.sidebar.slider(f"Weight f{i}", min_value=0.0, max_value=1.0, value=float(default_weights[i]), step=0.05))
    weights = np.array(weight_inputs)
    if weights.sum() == 0:
        weights = default_weights
    ref_input = st.sidebar.text_input("Reference point (comma-separated)", "")
    reference_point = None
    if ref_input.strip():
        try:
            reference_point = np.array([float(v.strip()) for v in ref_input.split(",")])
        except Exception:
            st.sidebar.error("Invalid reference point format.")
    method = st.sidebar.selectbox("MCDM method", ["tchebycheff", "weighted_sum", "knee", "topsis"])

    views = build_decision_views([primary_front], weights, reference_point, method)
    view = views[0]
    scores = view.mcdm_scores.get(method, np.zeros(view.front.points_F.shape[0]))
    order = np.argsort(scores)

    st.subheader("Pareto Front Comparison")
    obj_idx = st.multiselect("Objectives (choose 2)", list(range(primary_front.points_F.shape[1])), default=[0, 1])

    if len(obj_idx) == 2:
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
        fig = px.scatter(df, x=f"f{obj_idx[0]}", y=f"f{obj_idx[1]}", color="algorithm", title=f"{problem} Pareto Fronts")

        best_idx = int(order[0])
        best_point = primary_front.points_F[best_idx]
        fig.add_scatter(
            x=[best_point[obj_idx[0]]],
            y=[best_point[obj_idx[1]]],
            mode="markers",
            marker=dict(size=12, color="red", symbol="star", line=dict(width=2, color="black")),
            name=f"Best (Primary: {method})",
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"Top solutions ({primary_algo})")
    top_k = st.slider("Show top-k", min_value=1, max_value=min(20, len(order)), value=min(5, len(order)))
    top_indices = order[:top_k]
    st.write("Indices (Primary):", top_indices.tolist())

    import pandas as pd

    st.dataframe(pd.DataFrame(primary_front.points_F[top_indices], columns=[f"f{i}" for i in range(primary_front.points_F.shape[1])]))

    if st.button("Export top-k as JSON"):
        from vamos.ux.studio.export import export_solutions_to_json

        path = export_solutions_to_json(view, top_indices.tolist(), study_dir / "studio_export.json")
        st.success(f"Exported to {path}")

    st.subheader("Parallel Coordinates")
    try:
        fig_pc = px.parallel_coordinates(
            view.normalized_F,
            color=scores,
            labels={i: f"f{i}" for i in range(view.normalized_F.shape[1])},
        )
        st.plotly_chart(fig_pc, use_container_width=True)
    except Exception as exc:
        st.warning(f"Parallel coordinates unavailable: {exc}")

    st.subheader("Focused optimization")
    focus_budget = st.number_input("Budget (evaluations)", min_value=100, max_value=5000, value=500, step=100)
    if st.button("Run focused optimization"):
        if reference_point is None:
            st.error("Provide a reference point first.")
        else:
            with st.spinner("Running focused optimization..."):
                F_new, X_new = run_focused_optimization(problem, reference_point, primary_algo, int(focus_budget))
            st.success(f"Focused run produced {len(F_new)} points.")
            if len(obj_idx) == 2:
                fig2 = px.scatter(
                    x=F_new[:, obj_idx[0]],
                    y=F_new[:, obj_idx[1]],
                    labels={"x": f"f{obj_idx[0]}", "y": f"f{obj_idx[1]}"},
                )
                st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Search Dynamics (Re-run)")
    if primary_front.extra.get("config"):
        if st.button("Animate Optimization Evolution"):
            config = primary_front.extra["config"]
            with st.spinner("Re-running optimization to capture history..."):
                _, history = run_with_history(problem, config, int(focus_budget))

            if not history:
                st.warning("No history captured. Ensure algorithm supports live_viz/callbacks.")
            else:
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
    else:
        st.info("Config not available for this run (cannot re-run accurately).")


__all__ = ["main"]
