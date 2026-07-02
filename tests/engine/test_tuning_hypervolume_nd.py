from __future__ import annotations

import numpy as np

from vamos.engine.tuning.convergence import TuningCallback


def test_tuning_callback_records_three_objective_hypervolume() -> None:
    callback = TuningCallback([2.0, 2.0, 2.0])

    callback.on_generation(1, F=np.array([[1.0, 1.0, 1.0]], dtype=float))

    assert callback.hv_values == [1.0]


def test_tuning_callback_does_not_record_silent_zero_for_invalid_reference() -> None:
    callback = TuningCallback([0.5, 0.5, 0.5])

    callback.on_generation(1, F=np.array([[1.0, 1.0, 1.0]], dtype=float))

    assert callback.hv_values == []
