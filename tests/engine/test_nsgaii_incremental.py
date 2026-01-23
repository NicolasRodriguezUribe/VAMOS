import numpy as np

from vamos.engine.algorithm.nsgaii.helpers import (
    fronts_from_ranks,
    compute_crowding,
    incremental_insert_fronts,
    select_nsga2,
)
from vamos.foundation.kernel.numpy_backend import NumPyKernel


def test_incremental_survival_matches_full() -> None:
    rng = np.random.default_rng(123)
    pop_size = 30
    n_var = 4
    n_obj = 2

    X = rng.random((pop_size, n_var))
    F = rng.random((pop_size, n_obj))
    X_off = rng.random((1, n_var))
    F_off = rng.random((1, n_obj))

    kernel = NumPyKernel()
    X_full, F_full, sel_full = kernel.nsga2_survival(X, F, X_off, F_off, pop_size, return_indices=True)

    ranks, _ = kernel.nsga2_ranking(F)
    fronts = fronts_from_ranks(ranks)

    combined_X = np.vstack([X, X_off])
    combined_F = np.vstack([F, F_off])
    ranks_inc = np.concatenate([ranks, np.array([-1], dtype=int)])
    fronts_inc = [list(front) for front in fronts]
    incremental_insert_fronts(fronts_inc, ranks_inc, combined_F, combined_F.shape[0] - 1)
    crowding = compute_crowding(combined_F, fronts_inc)
    sel_inc = select_nsga2(fronts_inc, crowding, pop_size)
    X_inc = combined_X[sel_inc]
    F_inc = combined_F[sel_inc]

    assert np.array_equal(sel_full, sel_inc)
    assert np.allclose(X_full, X_inc)
    assert np.allclose(F_full, F_inc)
