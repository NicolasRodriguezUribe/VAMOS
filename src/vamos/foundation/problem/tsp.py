# problem/tsp.py
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

from .tsplib import load_tsplib_coords


def _default_coordinates() -> np.ndarray:
    # Simple layout with a few asymmetric points to avoid symmetry degeneracy.
    return np.array(
        [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.3, 0.8),
            (0.5, 1.5),
            (-0.3, 1.0),
            (-0.6, 0.2),
        ],
        dtype=float,
    )


def _circle_coordinates(n_cities: int) -> np.ndarray:
    if n_cities < 3:
        raise ValueError("TSP problems require at least 3 cities.")
    angles = np.linspace(0.0, 2.0 * math.pi, num=n_cities, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


class TSPProblem:
    """
    Toy multi-objective travelling salesman problem.
    Objective 1: total tour length (closed tour).
    Objective 2: maximum edge length in the tour (encourages balanced legs).
    """

    def __init__(
        self,
        n_cities: int | None = None,
        coordinates: Sequence[Sequence[float]] | None = None,
        dataset: str | None = None,
    ) -> None:
        if dataset is not None:
            coords = load_tsplib_coords(dataset)
        elif coordinates is not None:
            coords = np.asarray(coordinates, dtype=float)
        elif n_cities is not None:
            coords = _circle_coordinates(int(n_cities))
        else:
            coords = _default_coordinates()
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("Coordinates must be an array-like of shape (n_cities, 2).")
        self.coordinates = coords
        self.n_var = coords.shape[0]
        self.n_obj = 2
        self.xl = 0
        self.xu = self.n_var - 1
        self.encoding = "permutation"
        self.labels = [f"City {i}" for i in range(self.n_var)]

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        routes = np.asarray(X, dtype=int)
        if routes.ndim != 2 or routes.shape[1] != self.n_var:
            raise ValueError(f"Expected routes of shape (N, {self.n_var}).")
        coords = self.coordinates
        idx = np.mod(routes, self.n_var)
        paths = coords[idx]
        closed_paths = np.concatenate([paths, paths[:, :1, :]], axis=1)
        deltas = np.diff(closed_paths, axis=1)
        edges = np.linalg.norm(deltas, axis=2)
        F = out["F"]
        F[:, 0] = edges.sum(axis=1)
        F[:, 1] = edges.max(axis=1)

    def describe(self) -> dict[str, Iterable[str] | int]:
        return {
            "n_var": self.n_var,
            "cities": self.labels,
        }
