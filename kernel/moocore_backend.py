# kernel/moocore_backend.py
import numpy as np

try:
    import moocore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "MooCoreKernel requires the 'moocore' dependency. Install it or switch to a different backend."
    ) from exc

from .numpy_backend import NumPyKernel as _NumPyKernel, _compute_crowding, _select_nsga2


def _fronts_from_ranks(ranks: np.ndarray):
    if ranks.size == 0:
        return []
    unique_ranks = np.unique(ranks)
    return [np.flatnonzero(ranks == r).tolist() for r in unique_ranks]


class MooCoreKernel:
    """
    Backend that delegates non-dominated sorting to moocore (C implementation).
    The rest of the operators reuse the NumPy implementations.
    """

    @staticmethod
    def nsga2_ranking(F: np.ndarray):
        ranks = moocore.pareto_rank(np.asarray(F, dtype=np.float64, order="C"))
        fronts = _fronts_from_ranks(ranks)
        crowding = _compute_crowding(F, fronts)
        return ranks, crowding

    tournament_selection = staticmethod(_NumPyKernel.tournament_selection)
    sbx_crossover = staticmethod(_NumPyKernel.sbx_crossover)
    polynomial_mutation = staticmethod(_NumPyKernel.polynomial_mutation)

    @staticmethod
    def nsga2_survival(
        X: np.ndarray,
        F: np.ndarray,
        X_off: np.ndarray,
        F_off: np.ndarray,
        pop_size: int,
    ):
        X_comb = np.vstack((X, X_off))
        F_comb = np.vstack((F, F_off))
        ranks = moocore.pareto_rank(np.asarray(F_comb, dtype=np.float64, order="C"))
        fronts = _fronts_from_ranks(ranks)
        crowding = _compute_crowding(F_comb, fronts)
        sel = _select_nsga2(fronts, crowding, pop_size)
        return X_comb[sel], F_comb[sel]


class MooCoreKernelV2(MooCoreKernel):
    """
    Variant that leverages additional moocore helpers during survival
    to reduce the NumPy workload.
    """

    @staticmethod
    def nsga2_survival(
        X: np.ndarray,
        F: np.ndarray,
        X_off: np.ndarray,
        F_off: np.ndarray,
        pop_size: int,
    ):
        X_comb = np.vstack((X, X_off))
        F_comb = np.vstack((F, F_off))
        F_comb = np.asarray(F_comb, dtype=np.float64, order="C")
        ranks = moocore.pareto_rank(F_comb)

        keep_idx = []
        remaining = pop_size
        current_rank = 0
        while remaining > 0:
            idx = np.flatnonzero(ranks == current_rank)
            if idx.size == 0:
                current_rank += 1
                continue

            front_mask = np.zeros(F_comb.shape[0], dtype=bool)
            front_mask[idx] = True
            nondom_mask = moocore.is_nondominated(F_comb[front_mask])
            front_idx = idx[nondom_mask]

            if front_idx.size <= remaining:
                keep_idx.extend(front_idx.tolist())
                remaining -= front_idx.size
            else:
                ref_point = np.max(F_comb, axis=0) + 1.0
                hv_contrib = moocore.hv_contributions(
                    F_comb[front_idx], ref=ref_point
                )
                order = np.argsort(hv_contrib)[::-1]
                keep_idx.extend(front_idx[order[:remaining]].tolist())
                remaining = 0

            current_rank += 1

        keep = np.array(keep_idx[:pop_size], dtype=int)
        return X_comb[keep], F_comb[keep]
