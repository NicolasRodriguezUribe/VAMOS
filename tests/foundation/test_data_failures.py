import pytest

from vamos.foundation.data import reference_front_path, weight_path


def test_unknown_reference_front_errors():
    with pytest.raises(ValueError, match="Unknown reference front"):
        reference_front_path("does_not_exist")


def test_unknown_weight_errors():
    with pytest.raises(ValueError, match="Unknown weights file"):
        weight_path("missing.csv")
