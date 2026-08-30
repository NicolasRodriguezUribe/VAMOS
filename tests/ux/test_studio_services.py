from pathlib import Path

import vamos
from vamos.ux.studio.services import build_demo_study_data, discover_study_directories


def test_discover_study_directories_finds_results_roots(tmp_path: Path) -> None:
    run_dir = tmp_path / "results" / "quickstart" / "zdt1" / "nsgaii" / "seed_0"
    result = vamos.optimize("zdt1", pop_size=4, max_evaluations=4, seed=0)
    vamos.save_result(result, run_dir)

    study_dirs = discover_study_directories(tmp_path)

    assert tmp_path / "results" in study_dirs
    assert tmp_path / "results" / "quickstart" in study_dirs


def test_build_demo_study_data_marks_demo_fronts() -> None:
    runs, fronts = build_demo_study_data()

    assert len(runs) == 2
    assert len(fronts) == 2
    assert all(front.extra.get("demo") for front in fronts)
    assert fronts[0].points_F.shape[1] == 2
