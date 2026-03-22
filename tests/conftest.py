from __future__ import annotations

import numpy as np
import pytest

from vamos.foundation.problem.dtlz import DTLZ2Problem
from vamos.foundation.problem.zdt1 import ZDT1Problem


@pytest.fixture
def seeded_rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def zdt1_problem() -> ZDT1Problem:
    return ZDT1Problem(n_var=8)


@pytest.fixture
def dtlz2_problem() -> DTLZ2Problem:
    return DTLZ2Problem(n_var=12, n_obj=3)
