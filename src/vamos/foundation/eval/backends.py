from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, cast

import numpy as np

from vamos.foundation.eval.population import evaluate_population_with_constraints
from vamos.foundation.exceptions import ConfigurationError

from . import EvaluationBackend, EvaluationResult


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def _eval_chunk(problem: Any, X_chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
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

    def __init__(self, n_workers: int | None = None, chunk_size: int | None = None, timeout: float | None = None) -> None:
        self.n_workers = max(1, n_workers or os.cpu_count() or 1)
        self.chunk_size = chunk_size
        self.timeout = timeout

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
        G_parts: list[tuple[int, np.ndarray | None]] = []

        with ProcessPoolExecutor(max_workers=self.n_workers) as ex:
            future_map = {ex.submit(_eval_chunk, problem, X[start:end]): (start, end) for start, end in slices}
            for fut in as_completed(future_map):
                start, end = future_map[fut]
                F_chunk, G_chunk = fut.result(timeout=self.timeout)
                F_parts.append((start, F_chunk))
                G_parts.append((start, G_chunk))

        # Restore original order
        if not F_parts:
            raise RuntimeError(
                "MultiprocessingEvalBackend: no results were collected from worker chunks. "
                "All futures may have failed or the pool was empty."
            )
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
            raise ValueError(
                "MultiprocessingEvalBackend: one or more worker chunks returned no "
                "constraint data (G=None). Ensure all evaluations compute constraints."
            )

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
        self._logged_fallback = False
        self._owns_client = False

        try:
            from dask.distributed import Client

            if self.client is None:
                if self.address:
                    self.client = cast(Any, Client)(self.address)
                    self._owns_client = True
                else:
                    _logger().debug(
                        "DaskEvalBackend initialized without a client/address; it will fall back to serial until a client or address is provided."
                    )
            self._connected = True
        except ImportError:
            _logger().debug("DaskEvalBackend unavailable (missing dask.distributed); falling back to serial.")

    def close(self) -> None:
        """Close the Dask client if this backend created it."""
        if self._owns_client and self.client is not None:
            try:
                self.client.close()
            except Exception:
                _logger().debug("Error closing Dask client.", exc_info=True)
            finally:
                self.client = None
                self._connected = False

    def evaluate(self, X: np.ndarray, problem: Any) -> EvaluationResult:
        if not self._connected or (self.client is None and self.address is None):
            if not self._logged_fallback:
                _logger().warning("DaskEvalBackend not connected; falling back to SerialEvalBackend.")
                self._logged_fallback = True
            return SerialEvalBackend().evaluate(X, problem)

        try:
            # Re-check client connection
            if self.client is None and self.address:
                from dask.distributed import Client

                self.client = cast(Any, Client)(self.address)
                self._owns_client = True

            if self.client is None:
                # Fallback
                return SerialEvalBackend().evaluate(X, problem)

            n = X.shape[0]

            # Determine worker count with fallback if scheduler is unreachable
            try:
                n_workers = len(self.client.scheduler_info()["workers"])
            except Exception:
                n_workers = 1
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
            if not results:
                raise RuntimeError(
                    "DaskEvalBackend: no results returned from workers. All futures may have failed or the futures list was empty."
                )
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
                if G is not None:
                    if g_chunk is None:
                        raise ValueError(
                            "DaskEvalBackend: worker chunk returned no constraint data (G=None) "
                            "but earlier chunks did. Ensure all evaluations compute constraints."
                        )
                    G[start:end] = g_chunk

            return EvaluationResult(F=F, G=G)

        except Exception:
            _logger().warning("DaskEvalBackend evaluation failed; falling back to SerialEvalBackend.", exc_info=True)
            return SerialEvalBackend().evaluate(X, problem)


def resolve_eval_strategy(
    name: str, *, n_workers: int | None = None, chunk_size: int | None = None, dask_address: str | None = None
) -> EvaluationBackend:
    _KNOWN = ("serial", "multiprocessing", "dask")
    key = (name or "serial").lower()
    if key == "multiprocessing":
        return MultiprocessingEvalBackend(n_workers=n_workers, chunk_size=chunk_size)
    if key == "dask":
        return DaskEvalBackend(address=dask_address)
    if key != "serial":
        raise ConfigurationError(
            f"Unknown eval_strategy {name!r}.",
            suggestion=f"Valid strategies: {_KNOWN}.",
        )
    return SerialEvalBackend()


__all__ = [
    "EvaluationBackend",
    "SerialEvalBackend",
    "MultiprocessingEvalBackend",
    "DaskEvalBackend",
    "resolve_eval_strategy",
]
