"""
GCES algorithm entrypoint.

Phase 1 keeps GCES narrowly divergent from NSGA-II: the host loop, mating, and
front filling are reused, and only split-front truncation is delegated to the
GCES selector.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from vamos.engine.hooks.live_viz import LiveVisualization
from vamos.foundation.eval.backends import EvaluationBackend
from vamos.foundation.problem.types import ProblemProtocol

from ..nsgaii import NSGAII
from ..nsgaii.helpers import fronts_from_ranks
from ..nsgaii.state import update_archives
from . import selector as gces_selector


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def _prefilter_archive_candidates(
    st: Any,
    kernel: Any,
    X: np.ndarray,
    F: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Trim unconstrained archive updates to the combined nondominated set.

    This mirrors the NSGA-II archive prefilter locally so GCES does not depend
    on NSGA-II's private ask/tell helper surface.
    """
    if (st.archive_manager is None and st.result_archive is None) or F.shape[0] <= st.pop_size:
        return X, F

    try:
        ranks, _ = kernel.nsga2_ranking(F)
    except (ValueError, IndexError):
        _logger().debug("Failed to prefilter GCES archive candidates; using full batch", exc_info=True)
        return X, F

    nd_mask = ranks == ranks.min(initial=0)
    if bool(np.all(nd_mask)):
        return X, F
    return X[nd_mask], F[nd_mask]


class GCES(NSGAII):
    """
    Thin GCES wrapper around the NSGA-II runtime.

    Phase 1 keeps the public algorithm identity separate while reusing the
    existing NSGA-II run loop, ask path, and state management. The only
    algorithmic divergence is split-front truncation inside :meth:`tell`.
    """

    def _validate_phase1_support(self, problem: ProblemProtocol) -> None:
        n_constraints = int(getattr(problem, "n_constraints", 0) or 0)
        if n_constraints > 0:
            raise ValueError("GCES phase 1 does not support constrained problems.")

        if str(getattr(self.kernel, "name", "")).lower() == "moocore":
            raise ValueError("GCES phase 1 does not support the moocore engine.")

        if bool(self.cfg.get("track_genealogy", False)):
            raise ValueError("GCES phase 1 does not support genealogy tracking.")

        if bool(self.cfg.get("steady_state", False)):
            raise ValueError("GCES phase 1 does not support steady-state mode.")

        pop_size = int(self.cfg["pop_size"])
        offspring_size = int(self.cfg.get("offspring_size") or pop_size)
        if offspring_size < pop_size:
            raise ValueError("GCES phase 1 does not support incremental replacement (offspring_size < pop_size).")

    def _validate_phase1_state(self) -> None:
        st = self._st
        if st is None:
            raise RuntimeError("GCES state is not initialized.")
        if st.G is not None:
            raise ValueError("GCES phase 1 does not support constrained problems.")
        if bool(self.cfg.get("steady_state", False)):
            raise ValueError("GCES phase 1 does not support steady-state mode.")
        if bool(st.track_genealogy):
            raise ValueError("GCES phase 1 does not support genealogy tracking.")
        if bool(st.incremental_mode):
            raise ValueError("GCES phase 1 does not support steady-state or incremental replacement.")
        if str(getattr(self.kernel, "name", "")).lower() == "moocore":
            raise ValueError("GCES phase 1 does not support the moocore engine.")

    def run(
        self,
        problem: ProblemProtocol,
        termination: tuple[str, Any] = ("max_evaluations", 25000),
        seed: int = 0,
        eval_strategy: EvaluationBackend | None = None,
        live_viz: LiveVisualization | None = None,
        checkpoint_dir: str | None = None,
        checkpoint_interval: int = 50,
        checkpoint: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._validate_phase1_support(problem)
        return super().run(
            problem,
            termination,
            seed,
            eval_strategy=eval_strategy,
            live_viz=live_viz,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
            checkpoint=checkpoint,
        )

    def _select_split_front(
        self,
        F_split: np.ndarray,
        slots: int,
        ideal: np.ndarray,
        nadir: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return local survivor indices for the split front."""
        return gces_selector.select_split_front_gces(F_split, slots, ideal, nadir, rng)

    def tell(self, eval_result: Any, problem: ProblemProtocol | None = None) -> bool:
        """
        Apply GCES environmental selection for one generational NSGA-II update.

        The host flow stays standard: merge parent and offspring populations,
        rank the merged population, fill all complete fronts unchanged, and use
        GCES only when the next front must be truncated to fit the population.
        Normalization bounds passed to the selector are taken from the full
        merged objective matrix.
        """
        del problem
        st = self._st
        if st is None:
            raise RuntimeError("tell() called before initialization.")

        self._validate_phase1_state()

        X_off = st.pending_offspring
        st.pending_offspring = None
        if X_off is None:
            raise ValueError("tell() called without a pending ask().")

        F_off = np.asarray(eval_result.F)
        G_off = eval_result.G if st.constraint_mode != "none" else None
        if G_off is not None:
            raise ValueError("GCES phase 1 does not support constrained problems.")
        assert st.hv_tracker is not None

        combined_X = np.vstack([st.X, X_off])
        combined_F = np.vstack([st.F, F_off])

        ranks, _crowding = self.kernel.nsga2_ranking(combined_F)
        fronts = fronts_from_ranks(ranks)

        selected: list[int] = []
        ideal = np.min(combined_F, axis=0)
        nadir = np.max(combined_F, axis=0)

        # Reuse the usual NSGA-II front filling and delegate only the split
        # front truncation decision to the GCES selector.
        for front in fronts:
            if not front:
                continue
            front_arr = np.asarray(front, dtype=int)
            if len(selected) + front_arr.size <= st.pop_size:
                selected.extend(front_arr.tolist())
                continue

            slots = st.pop_size - len(selected)
            if slots > 0:
                local_idx = np.asarray(
                    self._select_split_front(
                        combined_F[front_arr],
                        slots,
                        ideal,
                        nadir,
                        st.rng,
                    ),
                    dtype=int,
                )
                if local_idx.ndim != 1:
                    raise ValueError("GCES selector must return a 1D index array.")
                if local_idx.size != slots:
                    raise ValueError("GCES selector returned an unexpected number of indices.")
                if local_idx.size and (
                    np.any(local_idx < 0)
                    or np.any(local_idx >= front_arr.size)
                    or np.unique(local_idx).size != local_idx.size
                ):
                    raise ValueError("GCES selector returned invalid split-front indices.")
                selected.extend(front_arr[local_idx].tolist())
            break

        selected_idx = np.asarray(selected, dtype=int)
        if selected_idx.size != st.pop_size:
            raise ValueError("GCES survival did not produce the expected population size.")

        new_X = combined_X[selected_idx]
        new_F = combined_F[selected_idx]
        new_G = None

        st.X, st.F, st.G = new_X, new_F, new_G
        st.pending_offspring_ids = None

        archive_X, archive_F = _prefilter_archive_candidates(
            st,
            self.kernel,
            combined_X,
            combined_F,
        )
        update_archives(st, self.kernel, X=archive_X, F=archive_F, G=None)

        return st.hv_tracker.enabled and st.hv_tracker.reached(st.hv_points_fn())


class GCESNoComp(GCES):
    """GCES ablation that disables component detection on the split front."""

    def _select_split_front(
        self,
        F_split: np.ndarray,
        slots: int,
        ideal: np.ndarray,
        nadir: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return gces_selector.select_split_front_gces_nocomp(F_split, slots, ideal, nadir, rng)


class GCESNoGeo(GCES):
    """GCES ablation that replaces geodesic with Euclidean farthest-first."""

    def _select_split_front(
        self,
        F_split: np.ndarray,
        slots: int,
        ideal: np.ndarray,
        nadir: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return gces_selector.select_split_front_gces_nogeo(F_split, slots, ideal, nadir, rng)


class NSGA2Farthest(GCES):
    """NSGA-II-hosted variant using deterministic farthest-first truncation."""

    def _select_split_front(
        self,
        F_split: np.ndarray,
        slots: int,
        ideal: np.ndarray,
        nadir: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return gces_selector.select_split_front_nsga2_farthest(F_split, slots, ideal, nadir, rng)


class NSGA2GapFill(GCES):
    """NSGA-II-hosted variant using deterministic 2D gap-filling truncation."""

    def _select_split_front(
        self,
        F_split: np.ndarray,
        slots: int,
        ideal: np.ndarray,
        nadir: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return gces_selector.select_split_front_nsga2_gapfill(F_split, slots, ideal, nadir, rng)


class NSGA2CurvGap(GCES):
    """NSGA-II-hosted variant using deterministic 2D curvature-aware gap filling."""

    def _select_split_front(
        self,
        F_split: np.ndarray,
        slots: int,
        ideal: np.ndarray,
        nadir: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return gces_selector.select_split_front_nsga2_curvgap(F_split, slots, ideal, nadir, rng)


class NSGA2HVFarthest(GCES):
    """NSGA-II-hosted variant using farthest-first plus hypervolume gain."""

    def _select_split_front(
        self,
        F_split: np.ndarray,
        slots: int,
        ideal: np.ndarray,
        nadir: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return gces_selector.select_split_front_nsga2_hvfarthest(F_split, slots, ideal, nadir, rng)


class NSGA2RefCoverFarthest(GCES):
    """NSGA-II-hosted variant using farthest-first plus reference-cover gain."""

    def _select_split_front(
        self,
        F_split: np.ndarray,
        slots: int,
        ideal: np.ndarray,
        nadir: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return gces_selector.select_split_front_nsga2_refcover_farthest(F_split, slots, ideal, nadir, rng)


class NSGA2HVRefFarthest(GCES):
    """NSGA-II-hosted variant mixing farthest-first, HV gain, and reference cover."""

    def _select_split_front(
        self,
        F_split: np.ndarray,
        slots: int,
        ideal: np.ndarray,
        nadir: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return gces_selector.select_split_front_nsga2_hvref_farthest(F_split, slots, ideal, nadir, rng)


class NSGA2SectorFarthest(GCES):
    """NSGA-II-hosted 3D variant using farthest-first plus sector rarity."""

    def _select_split_front(
        self,
        F_split: np.ndarray,
        slots: int,
        ideal: np.ndarray,
        nadir: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return gces_selector.select_split_front_nsga2_sector_farthest(F_split, slots, ideal, nadir, rng)


__all__ = [
    "GCES",
    "GCESNoComp",
    "GCESNoGeo",
    "NSGA2Farthest",
    "NSGA2GapFill",
    "NSGA2CurvGap",
    "NSGA2HVFarthest",
    "NSGA2RefCoverFarthest",
    "NSGA2HVRefFarthest",
    "NSGA2SectorFarthest",
]
