from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List, TYPE_CHECKING

import numpy as np

from vamos.ux.studio.runner import run_focused_optimization, run_with_history

if TYPE_CHECKING:
    from vamos.ux.studio.data import FrontRecord, RunRecord
    from vamos.ux.studio.dm import DecisionView


def _import_streamlit() -> Any:
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("VAMOS Studio requires the 'studio' extra: pip install -e \".[studio]\"") from exc
    return st


def _import_plotly() -> Any:
    try:
        import plotly.express as px  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Plotly is required for interactive plots. Install with the 'studio' extras.") from exc
    return px


def _load_data(study_dir: Path) -> tuple[list["RunRecord"], list["FrontRecord"]]:
    from vamos.ux.studio.data import load_runs_from_study, build_fronts

    runs = load_runs_from_study(study_dir)
    fronts = build_fronts(runs)
    return runs, fronts


def _build_decision_views(
    fronts: list["FrontRecord"],
    weights: np.ndarray,
    reference_point: np.ndarray | None,
    method: str,
) -> list["DecisionView"]:
    from vamos.ux.studio.dm import build_decision_view

    views = []
    for front in fronts:
        view = build_decision_view(front, weights=weights, reference_point=reference_point, methods=[method, "weighted_sum", "knee"])
        views.append(view)
    return views


def main(argv: List[str] | None = None) -> None:
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

    runs, fronts = _load_data(study_dir)
    st.sidebar.success(f"Loaded {len(runs)} runs across {len(fronts)} fronts.")

    problems = sorted({f.problem_name for f in fronts})
    problem = st.sidebar.selectbox("Problem", problems)
    algos = sorted({f.algorithm_name for f in fronts if f.problem_name == problem})

    # Comparison Mode
    selected_algos = st.sidebar.multiselect("Algorithms to Compare", algos, default=algos[:1])

    # Primary algorithm for MCDM
    primary_algo = st.sidebar.selectbox("Primary Algorithm (for MCDM)", selected_algos) if selected_algos else None

    if not selected_algos:
        st.info("Select at least one algorithm to visualize.")
        return

    # Filter fronts
    comparison_fronts = [f for f in fronts if f.problem_name == problem and f.algorithm_name in selected_algos]
    primary_front = next((f for f in fronts if f.problem_name == problem and f.algorithm_name == primary_algo), None)

    if not primary_front:
        st.error("Primary front not found.")
        return
    if primary_algo is None:
        st.error("Primary algorithm not selected.")
        return

    # --- MCDM Setup (on Primary) ---
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

    views = _build_decision_views([primary_front], weights, reference_point, method)
    view = views[0]
    scores = view.mcdm_scores.get(method, np.zeros(view.front.points_F.shape[0]))
    order = np.argsort(scores)

    # --- Visualization ---
    st.subheader("Pareto Front Comparison")
    obj_idx = st.multiselect("Objectives (choose 2)", list(range(primary_front.points_F.shape[1])), default=[0, 1])

    if len(obj_idx) == 2:
        # Build comparison plot
        import pandas as pd  # type: ignore[import-untyped]

        plot_data = []
        for f in comparison_fronts:
            for i in range(f.points_F.shape[0]):
                plot_data.append(
                    {
                        f"f{obj_idx[0]}": f.points_F[i, obj_idx[0]],
                        f"f{obj_idx[1]}": f.points_F[i, obj_idx[1]],
                        "Algorithm": f.algorithm_name,
                        "Index": i,
                    }
                )

        df_plot = pd.DataFrame(plot_data)

        fig = px.scatter(
            df_plot,
            x=f"f{obj_idx[0]}",
            y=f"f{obj_idx[1]}",
            color="Algorithm",
            hover_data=["Index"],
            title=f"Pareto Front Comparison: {problem}",
            symbol="Algorithm",
        )

        # Overlay MCDM selection from Primary
        best_idx = order[0]
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

    # Display table of values
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
            # Show quick scatter
            if len(obj_idx) == 2:
                fig2 = px.scatter(x=F_new[:, obj_idx[0]], y=F_new[:, obj_idx[1]], labels={"x": f"f{obj_idx[0]}", "y": f"f{obj_idx[1]}"})
                st.plotly_chart(fig2, use_container_width=True)

    # --- Search Dynamics ---
    st.subheader("Search Dynamics (Re-run)")
    if primary_front.extra.get("config"):
        if st.button("Animate Optimization Evolution"):
            config = primary_front.extra["config"]
            # Use same seed/budget from config if possible, or override budget for speed
            # Use 'focus_budget' from UI
            with st.spinner("Re-running optimization to capture history..."):
                _, history = run_with_history(problem, config, int(focus_budget))

            if not history:
                st.warning("No history captured. Ensure algorithm supports live_viz/callbacks.")
            else:
                st.success(f"Captured {len(history)} generations.")
                # Build dataframe for animation
                import pandas as pd

                frames = []
                for gen, F_gen in enumerate(history):
                    # Downsample if too large?
                    if len(F_gen) > 200:
                        # Just pick first 200 for viz performance
                        F_gen = F_gen[:200]
                    for i in range(len(F_gen)):
                        frames.append({"Generation": gen, f"f{obj_idx[0]}": F_gen[i, obj_idx[0]], f"f{obj_idx[1]}": F_gen[i, obj_idx[1]]})
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

    # --- Landscape Analysis ---
    st.subheader("Landscape Analysis")
    if st.button("Run Fitness Landscape Analysis"):
        from vamos.foundation.problem.registry import make_problem_selection

        with st.spinner("Sampling landscape (Random Walk)..."):
            selection = make_problem_selection(problem)
            prob_instance = selection.instantiate()
            import pandas as pd

            # Simple Random Walk
            n_steps = 500
            n_var = prob_instance.n_var
            # Assume continuous [0,1] or use bounds
            xl = getattr(prob_instance, "xl", np.zeros(n_var))
            xu = getattr(prob_instance, "xu", np.ones(n_var))

            current_x = np.random.uniform(xl, xu)
            walk_f: list[np.ndarray] = []
            step_size = 0.05 * (xu - xl)  # 5% step

            for _ in range(n_steps):
                # Evaluate
                out: dict[str, Any] = {}
                prob_instance.evaluate(current_x.reshape(1, -1), out)
                f_val = out["F"][0]
                walk_f.append(f_val)

                # Perturb
                perturb = np.random.uniform(-step_size, step_size)
                candidate = np.clip(current_x + perturb, xl, xu)
                current_x = candidate

            walk_f_arr = np.asarray(walk_f, dtype=float)
            # Take f1 for visualization if multi-objective
            y_series = walk_f_arr[:, 0]

            # Autocorrelation
            lags = range(1, 50)
            autocorr = [pd.Series(y_series).autocorr(lag=lag) for lag in lags]

            # Interpret
            corr_1 = autocorr[0]
            interpretation = "Rugged" if corr_1 < 0.5 else "Smooth"

            st.write(f"**Interpretation (f1):** {interpretation} (Lag-1 Autocorr: {corr_1:.2f})")

            # Plot
            fig_land = px.line(
                x=list(lags), y=autocorr, labels={"x": "Lag", "y": "Autocorrelation"}, title="Fitness Landscape Autocorrelation (f1)"
            )
            st.plotly_chart(fig_land, use_container_width=True)


if __name__ == "__main__":
    main()
