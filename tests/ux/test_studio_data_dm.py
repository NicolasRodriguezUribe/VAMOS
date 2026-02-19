from pathlib import Path

import numpy as np

from vamos.ux.studio.data import RunRecord, build_fronts, normalize_objectives
from vamos.ux.studio.dm import build_decision_view, feasible_indices, filter_by_objective_ranges, rank_by_score
from vamos.ux.studio.export import export_solutions_to_csv, export_solutions_to_json


def test_build_fronts_and_decision_view(tmp_path: Path):
    # Create fake run directories
    run_dir = tmp_path / "prob" / "algo" / "seed_0"
    run_dir.mkdir(parents=True)
    np.savetxt(run_dir / "FUN.csv", np.array([[0.1, 0.2], [0.2, 0.1]]), delimiter=",")
    runs = [RunRecord(None, "exp", "prob", "algo", 0, np.array([[0.1, 0.2]]), None)]
    runs.extend([RunRecord(None, "exp2", "prob", "algo", 1, np.array([[0.2, 0.1]]), None)])
    fronts = build_fronts(runs)
    assert len(fronts) == 1
    view = build_decision_view(fronts[0])
    assert view.normalized_F.shape == fronts[0].points_F.shape
    order = rank_by_score(view, "weighted_sum")
    assert order.shape[0] == fronts[0].points_F.shape[0]
    feasible = feasible_indices(view, max_violation=0.0)
    assert feasible.size == fronts[0].points_F.shape[0]
    filt = filter_by_objective_ranges(view, [(0.0, 0.15), (None, None)])
    assert filt.size >= 1


def test_export_helpers(tmp_path: Path):
    from vamos.ux.studio.data import FrontRecord

    front = FrontRecord(
        problem_name="prob",
        algorithm_name="algo",
        points_F=np.array([[0.1, 0.2], [0.2, 0.1]]),
        points_X=np.array([[1.0, 2.0], [3.0, 4.0]]),
        constraints=None,
        extra={},
    )
    view = build_decision_view(front)
    json_path = export_solutions_to_json(view, [0], tmp_path / "out.json")
    csv_path = export_solutions_to_csv(view, [0, 1], tmp_path / "out.csv")
    assert json_path.exists()
    assert csv_path.exists()
    assert json_path.read_text()
    assert csv_path.read_text()


def test_normalize_objectives():
    F = np.array([[1, 2], [3, 4]])
    norm = normalize_objectives(F)
    assert np.allclose(norm[0], [0.0, 0.0])
    assert np.allclose(norm[1], [1.0, 1.0])
