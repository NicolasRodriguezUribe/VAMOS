from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple
import numpy as np

ArchiveType = Literal["size_cap", "epsilon_grid", "hvc_prune", "hybrid"]
PrunePolicy = Literal["crowding", "hv_contrib", "random", "mc_hv_contrib"]


@dataclass(frozen=True)
class BoundedArchiveConfig:
    enabled: bool = True
    archive_type: ArchiveType = "size_cap"
    nondominated_only: bool = True

    size_cap: int = 200
    epsilon: float = 0.01  # for epsilon_grid (objective space)
    prune_policy: PrunePolicy = "crowding"

    # HV contribution pruning
    hv_ref_point: Optional[List[float]] = None
    hv_samples: int = 20000  # for Monte Carlo contributions (m>2 fallback)
    rng_seed: int = 0


@dataclass
class ArchiveUpdate:
    evals: int
    before: int
    after: int
    inserted: int
    pruned: int
    prune_reason: str


def pareto_nondominated_mask(F: np.ndarray) -> np.ndarray:
    """
    Minimization assumed.
    Returns boolean mask for nondominated points. O(n^2) broadcast.
    """
    n = F.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=bool)
    # dominates(a,b) if all(a<=b) and any(a<b)
    le = F[:, None, :] <= F[None, :, :]
    lt = F[:, None, :] < F[None, :, :]
    dom = np.all(le, axis=2) & np.any(lt, axis=2)
    dominated = np.any(dom, axis=0)
    return ~dominated


def crowding_distance(F: np.ndarray) -> np.ndarray:
    """
    Standard NSGA-style crowding distance, higher is better.
    """
    n, m = F.shape
    if n == 0:
        return np.array([], dtype=float)
    D = np.zeros(n, dtype=float)
    # For each objective
    for j in range(m):
        idx = np.argsort(F[:, j])
        D[idx[0]] = D[idx[-1]] = float("inf")
        fmin, fmax = F[idx[0], j], F[idx[-1], j]
        denom = (fmax - fmin) if fmax > fmin else 1.0
        for k in range(1, n - 1):
            D[idx[k]] += (F[idx[k + 1], j] - F[idx[k - 1], j]) / denom
    return D


def hv_contrib_2d(F: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Exact HV contribution for 2D minimization, using rectangle decomposition.
    Assumes points are nondominated.
    """
    n = F.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    # Sort by f1 ascending
    idx = np.argsort(F[:, 0])
    P = F[idx]
    contrib = np.zeros(n, dtype=float)

    # Total HV in 2D for ND set
    # hv = sum_i (x_{i+1}-x_i) * (ref_y - y_i) with x_0 = P0.x, x_{n}=ref_x, but careful:
    # In minimization, rectangles extend to ref. Using standard: integrate along x:
    # width = (next_x - curr_x), height = (ref_y - curr_y) with next_x = P[i+1].x, last next_x = ref_x
    xs = P[:, 0]
    ys = P[:, 1]
    next_x = np.concatenate([xs[1:], np.array([ref[0]])])
    widths = np.maximum(0.0, next_x - xs)
    heights = np.maximum(0.0, ref[1] - ys)
    total_hv = np.sum(widths * heights)

    # Contribution of point i: HV(P) - HV(P \ {i})
    # Compute by removing each point: O(n^2) but n is capped (archive pruning context).
    for t in range(n):
        Q = np.delete(P, t, axis=0)
        if Q.shape[0] == 0:
            hv_q = 0.0
        else:
            xs2 = Q[:, 0]
            ys2 = Q[:, 1]
            next_x2 = np.concatenate([xs2[1:], np.array([ref[0]])])
            widths2 = np.maximum(0.0, next_x2 - xs2)
            heights2 = np.maximum(0.0, ref[1] - ys2)
            hv_q = float(np.sum(widths2 * heights2))
        contrib[t] = total_hv - hv_q

    # Undo sort
    out = np.zeros(n, dtype=float)
    out[idx] = contrib
    return out


class BoundedArchive:
    """
    Stores (X,F) pairs optionally; pruning is objective-space driven.
    Minimization assumed.
    """

    def __init__(self, cfg: BoundedArchiveConfig):
        if cfg.size_cap <= 0:
            raise ValueError("size_cap must be > 0")
        self.cfg = cfg
        self.X: Optional[np.ndarray] = None
        self.F: np.ndarray = np.zeros((0, 0), dtype=float)
        self._rng = np.random.default_rng(cfg.rng_seed)
        self.total_inserted = 0
        self.total_pruned = 0

    def size(self) -> int:
        return int(self.F.shape[0])

    def add(self, X: Optional[np.ndarray], F: np.ndarray, evals: int) -> ArchiveUpdate:
        before = self.size()
        F = np.asarray(F, dtype=float)
        if F.ndim != 2:
            raise ValueError("F must be 2D array (n,m)")
        if before == 0:
            self.F = F.copy()
            self.X = None if X is None else np.asarray(X).copy()
        else:
            # concat
            self.F = np.vstack([self.F, F])
            if X is not None:
                Xn = np.asarray(X)
                self.X = Xn.copy() if self.X is None else np.vstack([self.X, Xn])

        inserted = int(F.shape[0])
        self.total_inserted += inserted

        prune_reason = "none"
        # Optionally keep only nondominated
        if self.cfg.nondominated_only and self.size() > 1:
            mask = pareto_nondominated_mask(self.F)
            pruned_nd = int(np.sum(~mask))
            if pruned_nd > 0:
                self.F = self.F[mask]
                if self.X is not None:
                    self.X = self.X[mask]
                self.total_pruned += pruned_nd
                prune_reason = "dominance"

        # Apply bounding
        if self.size() > self.cfg.size_cap:
            pruned, reason = self._prune_to_cap()
            self.total_pruned += pruned
            prune_reason = reason

        after = self.size()
        pruned_total = max(0, before + inserted - after)
        return ArchiveUpdate(
            evals=evals,
            before=before,
            after=after,
            inserted=inserted,
            pruned=pruned_total,
            prune_reason=prune_reason,
        )

    def _prune_to_cap(self) -> Tuple[int, str]:
        n = self.size()
        cap = self.cfg.size_cap
        if n <= cap:
            return 0, "none"

        # epsilon grid compaction (keeps one rep per cell, then cap-prunes)
        if self.cfg.archive_type in ("epsilon_grid", "hybrid"):
            self._apply_epsilon_grid()

        # If still above cap, prune by policy
        n = self.size()
        if n <= cap:
            return 0, "epsilon_grid"

        policy = self.cfg.prune_policy
        if policy == "crowding":
            D = crowding_distance(self.F)
            # prune smallest distances
            order = np.argsort(D)  # ascending
            kill = order[: (n - cap)]
            keep = np.ones(n, dtype=bool)
            keep[kill] = False
            self.F = self.F[keep]
            if self.X is not None:
                self.X = self.X[keep]
            return int(np.sum(~keep)), "crowding"

        if policy in ("hv_contrib", "mc_hv_contrib"):
            m = self.F.shape[1]
            ref = self._ref_point(m)
            if m == 2:
                # ensure ND for contribution meaning
                if self.cfg.nondominated_only:
                    mask = pareto_nondominated_mask(self.F)
                    self.F = self.F[mask]
                    if self.X is not None:
                        self.X = self.X[mask]
                    n = self.size()
                contrib = hv_contrib_2d(self.F, ref=np.asarray(ref, dtype=float))
                # prune smallest contribution
                kill = np.argsort(contrib)[: (n - cap)]
            else:
                # Monte Carlo approx contributions for m>2
                kill = self._mc_hv_contrib_prune(ref, n - cap)

            keep = np.ones(self.size(), dtype=bool)
            keep[kill] = False
            self.F = self.F[keep]
            if self.X is not None:
                self.X = self.X[keep]
            return int(np.sum(~keep)), policy

        # random fallback
        kill = self._rng.choice(n, size=(n - cap), replace=False)
        keep = np.ones(n, dtype=bool)
        keep[kill] = False
        self.F = self.F[keep]
        if self.X is not None:
            self.X = self.X[keep]
        return int(np.sum(~keep)), "random"

    def _apply_epsilon_grid(self) -> None:
        eps = float(self.cfg.epsilon)
        if eps <= 0:
            return
        F = self.F
        # grid key by floor(F/eps)
        key = np.floor(F / eps).astype(int)
        # keep first occurrence per cell (can be improved: keep best by dominance/crowding)
        # stable unique
        _, idx = np.unique(key, axis=0, return_index=True)
        idx = np.sort(idx)
        self.F = F[idx]
        if self.X is not None:
            self.X = self.X[idx]

    def _ref_point(self, m: int) -> List[float]:
        if self.cfg.hv_ref_point is not None:
            if len(self.cfg.hv_ref_point) != m:
                raise ValueError("hv_ref_point dimension mismatch")
            return list(map(float, self.cfg.hv_ref_point))
        # conservative: ref = max(F) + 1 in each dim
        mx = np.max(self.F, axis=0) if self.size() else np.ones((m,))
        return [float(x + 1.0) for x in mx]

    def _mc_hv_contrib_prune(self, ref: List[float], k: int) -> np.ndarray:
        # Very simple Monte Carlo contribution proxy:
        # sample points uniformly in bounding box [min(F), ref], count dominated volume per point.
        # Not a full HV decomposition, but gives a monotone proxy for contributions under cap pruning.
        F = self.F
        n, m = F.shape
        refv = np.asarray(ref, dtype=float)

        lo = np.min(F, axis=0)
        hi = refv
        # Avoid degenerate boxes
        span = np.maximum(hi - lo, 1e-12)

        S = int(self.cfg.hv_samples)
        U = self._rng.random((S, m))
        samples = lo + U * span  # uniform in box

        # dominated_by_i: sample >= Fi in all dims (minimization rectangles to ref)
        # count per i
        counts = np.zeros(n, dtype=float)
        for i in range(n):
            dom = np.all(samples >= F[i], axis=1)
            counts[i] = np.sum(dom)

        # smaller dominated count => smaller contribution (proxy)
        kill = np.argsort(counts)[:k]
        return kill
