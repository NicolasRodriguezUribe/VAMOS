from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

def _import_streamlit():
    try:
        import streamlit as st  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "VAMOS Studio requires the 'studio' extra: pip install -e \".[studio]\""
        ) from exc
    return st


def _import_plotly():
    try:
        import plotly.express as px  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Plotly is required for interactive plots. Install with the 'studio' extras."
        ) from exc
    return px


def _load_data(study_dir: Path):
    from vamos.studio.data import load_runs_from_study, build_fronts

    runs = load_runs_from_study(study_dir)
    fronts = build_fronts(runs)
    return runs, fronts


def _build_decision_views(fronts, weights, reference_point, method):
    from vamos.studio.dm import build_decision_view

    views = []
    for front in fronts:
        view = build_decision_view(front, weights=weights, reference_point=reference_point, methods=[method, "weighted_sum", "knee"])
        views.append(view)
    return views


def _run_focused_optimization(problem: str, reference_point: np.ndarray, algo: str, budget: int):
    # Minimal focused re-run leveraging optimize(); uses reference point to bias ranking via Tchebycheff scores post-hoc.
    from vamos.problem.registry import make_problem_selection
    from vamos.core.optimize import optimize
    from vamos.algorithm.config import NSGAIIConfig
    from vamos.core.experiment_config import ExperimentConfig

    selection = make_problem_selection(problem)
    cfg = (
        NSGAIIConfig()
        .pop_size(40)
        .offspring_size(40)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", pressure=2)
        .survival("nsga2")
        .engine("numpy")
        .result_mode("population")
        .fixed()
    ).to_dict()
    # Override budget
    exp_cfg = ExperimentConfig(max_evaluations=budget)
    result = optimize(selection.instantiate(), cfg, termination=("n_eval", budget), seed=0, engine="numpy")
    F = result.F
    # Re-rank by distance to reference point
    from vamos.analysis.mcdm import reference_point_scores
    scores = reference_point_scores(F, reference_point).scores
    order = np.argsort(scores)
    return F[order], result.X[order] if result.X is not None else None


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
    algo = st.sidebar.selectbox("Algorithm", algos)

    front = next(f for f in fronts if f.problem_name == problem and f.algorithm_name == algo)
    default_weights = np.ones(front.points_F.shape[1]) / front.points_F.shape[1]
    weight_inputs = []
    st.sidebar.subheader("Preferences")
    for i in range(front.points_F.shape[1]):
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
    method = st.sidebar.selectbox("MCDM method", ["tchebycheff", "weighted_sum", "knee"])

    views = _build_decision_views([front], weights, reference_point, method)
    view = views[0]
    scores = view.mcdm_scores.get(method, np.zeros(view.front.points_F.shape[0]))
    order = np.argsort(scores)

    st.subheader("Pareto Front")
    obj_idx = st.multiselect("Objectives (choose 2)", list(range(front.points_F.shape[1])), default=[0, 1])
    if len(obj_idx) == 2:
        fig = px.scatter(
            x=front.points_F[:, obj_idx[0]],
            y=front.points_F[:, obj_idx[1]],
            color=scores,
            labels={"x": f"f{obj_idx[0]}", "y": f"f{obj_idx[1]}", "color": "score"},
            hover_name=[str(i) for i in range(front.points_F.shape[0])],
        )
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Top solutions")
    top_k = st.slider("Show top-k", min_value=1, max_value=min(20, len(order)), value=min(5, len(order)))
    top_indices = order[:top_k]
    st.write("Indices:", top_indices.tolist())
    if st.button("Export top-k as JSON"):
        from vamos.studio.export import export_solutions_to_json
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
                F_new, X_new = _run_focused_optimization(problem, reference_point, algo, int(focus_budget))
            st.success(f"Focused run produced {len(F_new)} points.")
            # Show quick scatter
            if len(obj_idx) == 2:
                fig2 = px.scatter(x=F_new[:, obj_idx[0]], y=F_new[:, obj_idx[1]], labels={"x": f"f{obj_idx[0]}", "y": f"f{obj_idx[1]}"})
                st.plotly_chart(fig2, use_container_width=True)


if __name__ == "__main__":
    main()
