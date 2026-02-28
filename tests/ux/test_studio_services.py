from pathlib import Path

import numpy as np

from vamos.ux.studio.services import build_demo_study_data, discover_study_directories


def test_discover_study_directories_finds_results_roots(tmp_path: Path) -> None:
    run_dir = tmp_path / "results" / "quickstart" / "zdt1" / "nsgaii" / "seed_0"
    run_dir.mkdir(parents=True)
    np.savetxt(run_dir / "FUN.csv", np.array([[0.1, 0.2]]), delimiter=",")

    study_dirs = discover_study_directories(tmp_path)

    assert tmp_path / "results" in study_dirs
    assert tmp_path / "results" / "quickstart" in study_dirs


def test_build_demo_study_data_marks_demo_fronts() -> None:
    runs, fronts = build_demo_study_data()

    assert len(runs) == 2
    assert len(fronts) == 2
    assert all(front.extra.get("demo") for front in fronts)
    assert fronts[0].points_F.shape[1] == 2
