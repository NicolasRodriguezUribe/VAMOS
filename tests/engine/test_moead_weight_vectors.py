from __future__ import annotations

import numpy as np
import pytest

from vamos import optimize
from vamos.engine.algorithm.components.weight_vectors import load_or_generate_weight_vectors
from vamos.foundation.problem.dtlz import DTLZ2Problem


def test_moead_rejects_incompatible_das_dennis_lattice() -> None:
    with pytest.raises(ValueError, match="pop_size=40|requires pop_size"):
        load_or_generate_weight_vectors(40, 3, divisions=8, mode="jmetalpy")


def test_moead_accepts_exact_das_dennis_lattice() -> None:
    weights = load_or_generate_weight_vectors(45, 3, divisions=8, mode="jmetalpy")

    assert weights.shape == (45, 3)
    assert np.any(np.all(np.isclose(weights, [1.0, 0.0, 0.0]), axis=1))
    assert np.any(np.all(np.isclose(weights, [0.0, 1.0, 0.0]), axis=1))
    assert np.any(np.all(np.isclose(weights, [0.0, 0.0, 1.0]), axis=1))


def test_moead_weight_file_must_match_pop_size_exactly(tmp_path) -> None:
    path = tmp_path / "W3D_4.dat"
    np.savetxt(path, np.full((3, 3), 1.0 / 3.0))

    with pytest.raises(ValueError, match="Expected 4 weight vectors"):
        load_or_generate_weight_vectors(4, 3, path=str(path), mode="jmetalpy")


def test_optimize_moead_default_pop_size_is_lattice_compatible() -> None:
    problem = DTLZ2Problem(n_var=7, n_obj=3)

    result = optimize(problem, algorithm="moead", max_evaluations=91, seed=1, engine="numpy")

    assert result.meta["resolved_config"]["pop_size"] == 91
    assert result.data["evaluations"] == 91


def test_optimize_moead_rejects_incompatible_explicit_pop_size() -> None:
    problem = DTLZ2Problem(n_var=7, n_obj=3)

    with pytest.raises(ValueError, match="Das-Dennis|pop_size"):
        optimize(problem, algorithm="moead", pop_size=40, max_evaluations=40, seed=1, engine="numpy")
