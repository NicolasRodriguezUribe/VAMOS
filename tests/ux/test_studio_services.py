from pathlib import Path

import vamos
from vamos.ux.studio.services import build_demo_study_data, discover_study_directories, load_studio_data


def test_discover_study_directories_finds_canonical_roots(tmp_path: Path) -> None:
    study_dir = tmp_path / "results" / "canonical"
    vamos.create_study(
        vamos.StudySpec(
            problems=["zdt1"],
            algorithms=["nsgaii"],
            seeds=[0],
            pop_size=4,
            max_evaluations=4,
            engine="numpy",
        ),
        output=study_dir,
    )

    study_dirs = discover_study_directories(tmp_path)

    assert study_dirs == [study_dir]


def test_load_studio_data_uses_summary_run_traceability(tmp_path: Path) -> None:
    study = vamos.create_study(
        vamos.StudySpec(
            problems=["zdt1"],
            algorithms=["nsgaii"],
            seeds=[0],
            pop_size=4,
            max_evaluations=4,
            engine="numpy",
        ),
        output=tmp_path / "study",
    ).run()

    runs, fronts = load_studio_data(study.root)

    assert len(runs) == len(fronts) == 1
    trace = runs[0].metadata["study_summary"]
    assert trace["study_id"] == study.study_id
    assert trace["selected_run_id"] == runs[0].experiment_id


def test_build_demo_study_data_marks_demo_fronts() -> None:
    runs, fronts = build_demo_study_data()

    assert len(runs) == 2
    assert len(fronts) == 2
    assert all(front.extra.get("demo") for front in fronts)
    assert fronts[0].points_F.shape[1] == 2
