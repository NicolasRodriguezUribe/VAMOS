from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Optional, cast

import numpy as np

from vamos.foundation.eval.population import evaluate_population_with_constraints
from . import EvaluationBackend, EvaluationResult


def _eval_chunk(problem: Any, X_chunk: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Worker helper to evaluate a chunk; kept at module level for pickling."""
    F, G = evaluate_population_with_constraints(problem, X_chunk)
    return F, G


class SerialEvalBackend(EvaluationBackend):
    """Synchronous in-process evaluation (current default)."""

    def evaluate(self, X: np.ndarray, problem: Any) -> EvaluationResult:
        F, G = evaluate_population_with_constraints(problem, X)
        return EvaluationResult(F=F, G=G)


class MultiprocessingEvalBackend(EvaluationBackend):
    """
    Parallel evaluation using multiprocessing.

    Notes:
        - Requires the problem instance to be picklable.
        - Best suited for expensive evaluations; overhead dominates for tiny problems.
    """

    def __init__(self, n_workers: Optional[int] = None, chunk_size: Optional[int] = None) -> None:
        self.n_workers = max(1, n_workers or os.cpu_count() or 1)
        self.chunk_size = chunk_size

    def evaluate(self, X: np.ndarray, problem: Any) -> EvaluationResult:
        if self.n_workers <= 1 or X.shape[0] <= 1:
            return SerialEvalBackend().evaluate(X, problem)

        n = X.shape[0]
        if self.chunk_size is not None and self.chunk_size > 0:
            chunk_size = self.chunk_size
        else:
            chunk_size = max(1, math.ceil(n / self.n_workers))
        slices = [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]

        F_parts: list[tuple[int, np.ndarray]] = []
        G_parts: list[tuple[int, Optional[np.ndarray]]] = []

        with ProcessPoolExecutor(max_workers=self.n_workers) as ex:
            future_map = {ex.submit(_eval_chunk, problem, X[start:end]): (start, end) for start, end in slices}
            for fut in as_completed(future_map):
                start, end = future_map[fut]
                F_chunk, G_chunk = fut.result()
                F_parts.append((start, F_chunk))
                G_parts.append((start, G_chunk))

        # Restore original order
        F = np.empty((n, F_parts[0][1].shape[1]), dtype=float)
        G_sample = G_parts[0][1]
        G_out: np.ndarray | None = None
        if G_sample is not None:
            G_out = np.empty((n, G_sample.shape[1]), dtype=float)
        for start, f_part in sorted(F_parts, key=lambda p: p[0]):
            F[start : start + f_part.shape[0]] = f_part
        missing_constraints = False
        if G_out is not None:
            for start, g_part in sorted(G_parts, key=lambda p: p[0]):
                if g_part is None:
                    missing_constraints = True
                    break
                G_out[start : start + g_part.shape[0]] = g_part
        if missing_constraints:
            G_out = None

        return EvaluationResult(F=F, G=G_out)


class DaskEvalBackend(EvaluationBackend):
    """
    Distributed evaluation using Dask.

    Notes:
        - Requires `dask.distributed`.
        - Falls back to serial if dask is not installed or client is invalid.
    """

    def __init__(self, client: Any = None, address: str | None = None) -> None:
        """
        Initialize Dask backend.

        Args:
            client: Existing dask.distributed.Client
            address: Address of scheduler to connect to (if client is None)
        """
        self.client = client
        self.address = address
        self._connected = False

        try:
            from dask.distributed import Client

            if self.client is None:
                if self.address:
                    self.client = cast(Any, Client)(self.address)
                else:
                    # Create local cluster if neither provided?
                    # Or let user manage strictness?
                    # For now, require explicit client or address for "remote",
                    # otherwise create LocalCluster implicitly?
                    # Better to defer creation to first evaluate call or let user pass it.
                    pass
            self._connected = True
        except ImportError:
            pass

    def evaluate(self, X: np.ndarray, problem: Any) -> EvaluationResult:
        if not self._connected or (self.client is None and self.address is None):
            return SerialEvalBackend().evaluate(X, problem)

        try:
            # Re-check client connection
            if self.client is None and self.address:
                from dask.distributed import Client

                self.client = cast(Any, Client)(self.address)

            if self.client is None:
                # Fallback
                return SerialEvalBackend().evaluate(X, problem)

            n = X.shape[0]
            # Map futures
            # Dask handles efficient chunking usually, but for strict comparison
            # with multiprocessing, we can chunk manually or let dask decide.
            # Simple map over rows:
            # futures = self.client.map(lambda x: _eval_chunk(problem, x[None, :]), X)

            # Better: chunk it to reduce overhead
            n_workers = len(self.client.scheduler_info()["workers"])
            chunk_size = max(1, math.ceil(n / n_workers))
            slices = [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]

            futures = []
            for start, end in slices:
                # Submit chunk
                fut = self.client.submit(_eval_chunk, problem, X[start:end])
                futures.append((start, fut))

            # Gather
            results = self.client.gather([f for _, f in futures])

            # Reassemble
            # Assuming first result gives dimensions
            F_sample = results[0][0]
            G_sample = results[0][1]

            F = np.empty((n, F_sample.shape[1]), dtype=float)
            G = None
            if G_sample is not None:
                G = np.empty((n, G_sample.shape[1]), dtype=float)

            for i, (start, _) in enumerate(futures):
                f_chunk, g_chunk = results[i]
                end = start + f_chunk.shape[0]
                F[start:end] = f_chunk
                if G is not None and g_chunk is not None:
                    G[start:end] = g_chunk

            return EvaluationResult(F=F, G=G)

        except Exception:
            # Fallback on failure
            return SerialEvalBackend().evaluate(X, problem)


def resolve_eval_strategy(
    name: str, *, n_workers: Optional[int] = None, chunk_size: Optional[int] = None, dask_address: str | None = None
) -> EvaluationBackend:
    key = (name or "serial").lower()
    if key == "multiprocessing":
        return MultiprocessingEvalBackend(n_workers=n_workers, chunk_size=chunk_size)
    if key == "dask":
        return DaskEvalBackend(address=dask_address)
    return SerialEvalBackend()


__all__ = [
    "EvaluationBackend",
    "SerialEvalBackend",
    "MultiprocessingEvalBackend",
    "DaskEvalBackend",
    "resolve_eval_strategy",
]
