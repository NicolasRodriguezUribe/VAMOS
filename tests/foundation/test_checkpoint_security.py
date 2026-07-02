from __future__ import annotations

import numpy as np
import pytest

from vamos.foundation.checkpoint import load_checkpoint, save_checkpoint


def test_load_checkpoint_requires_trusted_flag(tmp_path) -> None:
    rng = np.random.default_rng(0)
    path = save_checkpoint(
        tmp_path / "state",
        X=np.zeros((2, 3)),
        F=np.zeros((2, 2)),
        generation=1,
        n_eval=2,
        rng_state=rng.bit_generator.state,
    )

    with pytest.raises(ValueError, match="trusted=True"):
        load_checkpoint(path)

    loaded = load_checkpoint(path, trusted=True)
    assert loaded["n_eval"] == 2
    assert loaded["generation"] == 1
