from importlib import resources

from vamos.foundation.data import reference_front_path, weight_path
from vamos.foundation.core.hv_stop import build_hv_stop_config
from vamos.experiment.execution import _default_weight_path


def test_reference_front_packaged_and_accessible():
    path = reference_front_path("zdt1")
    assert path.is_file()
    data = path.read_text().splitlines()
    assert len(data) > 0


def test_weight_file_packaged_and_accessible():
    path = weight_path("zdt1problem_2obj_pop100.csv")
    assert path.is_file()
    assert path.stat().st_size > 0
    resolved = _default_weight_path("zdt1problem", 2, 100)
    assert resolved.endswith("zdt1problem_2obj_pop100.csv")


def test_hv_stop_uses_packaged_reference_front():
    cfg = build_hv_stop_config(0.1, None, "zdt1")
    assert cfg is not None
    ref_path = cfg["reference_front_path"]
    assert "ZDT1" in ref_path.upper()
    assert resources.files("vamos.foundation.data.reference_fronts").joinpath("ZDT1.csv").is_file()
