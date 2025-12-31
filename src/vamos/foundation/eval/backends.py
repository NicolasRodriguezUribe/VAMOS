from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import numpy as np

from vamos.foundation.eval.population import evaluate_population_with_constraints
from . import EvaluationBackend, EvaluationResult


def _eval_chunk(problem, X_chunk: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Worker helper to evaluate a chunk; kept at module level for pickling."""
    F, G = evaluate_population_with_constraints(problem, X_chunk)
    return F, G


class SerialEvalBackend(EvaluationBackend):
    """Synchronous in-process evaluation (current default)."""

    def evaluate(self, X: np.ndarray, problem) -> EvaluationResult:
        F, G = evaluate_population_with_constraints(problem, X)
        return EvaluationResult(F=F, G=G)


class MultiprocessingEvalBackend(EvaluationBackend):
    """
    Parallel evaluation using multiprocessing.

    Notes:
        - Requires the problem instance to be picklable.
        - Best suited for expensive evaluations; overhead dominates for tiny problems.
    """

    def __init__(self, n_workers: Optional[int] = None, chunk_size: Optional[int] = None):
        self.n_workers = max(1, n_workers or os.cpu_count() or 1)
        self.chunk_size = chunk_size

    def evaluate(self, X: np.ndarray, problem) -> EvaluationResult:
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
            future_map = {
                ex.submit(_eval_chunk, problem, X[start:end]): (start, end)
                for start, end in slices
            }
            for fut in as_completed(future_map):
                start, end = future_map[fut]
                F_chunk, G_chunk = fut.result()
                F_parts.append((start, F_chunk))
                G_parts.append((start, G_chunk))

        # Restore original order
        F = np.empty((n, F_parts[0][1].shape[1]), dtype=float)
        G_sample = G_parts[0][1]
        G = None
        if G_sample is not None:
            G = np.empty((n, G_sample.shape[1]), dtype=float)
        for start, part in sorted(F_parts, key=lambda p: p[0]):
            F[start : start + part.shape[0]] = part
        for start, part in sorted(G_parts, key=lambda p: p[0]):
            if part is None or G is None:
                G = None
                break
            G[start : start + part.shape[0]] = part

        return EvaluationResult(F=F, G=G)


def resolve_eval_backend(name: str, *, n_workers: Optional[int] = None, chunk_size: Optional[int] = None) -> EvaluationBackend:
    key = (name or "serial").lower()
    if key == "multiprocessing":
        return MultiprocessingEvalBackend(n_workers=n_workers, chunk_size=chunk_size)
    return SerialEvalBackend()


__all__ = ["SerialEvalBackend", "MultiprocessingEvalBackend", "resolve_eval_backend"]
