from __future__ import annotations

from typing import TypeAlias, cast

import numpy as np
import numpy.typing as npt

from .fshape import evaluate_f
from .gz import evaluate_g, zbias, zcat_evaluate_z, zcat_get_j, zcat_value_in

_COMPLICATED_G_FUNCTION_BY_PROBLEM = {
    1: 4,
    2: 5,
    3: 2,
    4: 7,
    5: 9,
    6: 4,
    7: 5,
    8: 2,
    9: 7,
    10: 9,
    11: 3,
    12: 10,
    13: 1,
    14: 6,
    15: 8,
    16: 10,
    17: 1,
    18: 8,
    19: 6,
    20: 3,
}

_ONE_DIMENSIONAL_PARETO_SET_PROBLEMS = (14, 15, 16)
_DOUBLE_MIN_VALUE = 5e-324
FloatArray: TypeAlias = npt.NDArray[np.float64]
FloatMatrix: TypeAlias = npt.NDArray[np.float64]


def _zeros(size: int) -> FloatArray:
    return cast(FloatArray, np.zeros(size, dtype=float))


class ZCATProblem:
    """
    NumPy implementation of the ZCAT benchmark suite (ZCAT1-ZCAT20).

    Reference:
    Zapotecas-Martinez et al., "Challenging test problems for multi-and many-objective optimization",
    DOI: 10.1016/j.swevo.2023.101350
    """

    def __init__(
        self,
        problem_id: int,
        *,
        n_var: int = 30,
        n_obj: int = 2,
        complicated_pareto_set: bool = False,
        level: int = 1,
        bias: bool = False,
        imbalance: bool = False,
    ) -> None:
        if problem_id < 1 or problem_id > 20:
            raise ValueError("ZCAT problem id must be in [1, 20].")
        if n_obj < 2:
            raise ValueError("ZCAT requires at least two objectives.")
        if n_var <= 0:
            raise ValueError("ZCAT requires n_var > 0.")

        min_required_n_var = 1 if problem_id in _ONE_DIMENSIONAL_PARETO_SET_PROBLEMS else n_obj - 1
        if n_var < min_required_n_var:
            raise ValueError(
                f"ZCAT{problem_id} requires n_var >= {min_required_n_var} for n_obj={n_obj}. Received n_var={n_var}."
            )

        self.problem_id = int(problem_id)
        self.n_var = int(n_var)
        self.n_obj = int(n_obj)
        self.complicated_pareto_set = bool(complicated_pareto_set)
        self.level = int(level)
        self.bias = bool(bias)
        self.imbalance = bool(imbalance)
        self.encoding = "continuous"
        self.name = f"zcat{self.problem_id}"

        indices: FloatArray = np.arange(1.0, self.n_var + 1.0, dtype=float)
        self.xl = -0.5 * indices
        self.xu = 0.5 * indices

    def _pareto_set_dimension(self, y0: float) -> int:
        if self.problem_id in _ONE_DIMENSIONAL_PARETO_SET_PROBLEMS:
            return 1

        if self.problem_id == 19:
            if zcat_value_in(y0, 0.0, 0.2) or zcat_value_in(y0, 0.4, 0.6):
                return 1
            return self.n_obj - 1

        if self.problem_id == 20:
            if zcat_value_in(y0, 0.1, 0.4) or zcat_value_in(y0, 0.6, 0.9):
                return 1
            return self.n_obj - 1

        return self.n_obj - 1

    def _g_function_id(self) -> int:
        if not self.complicated_pareto_set:
            return 0
        return _COMPLICATED_G_FUNCTION_BY_PROBLEM[self.problem_id]

    def _alpha(self, y: FloatArray) -> FloatArray:
        alpha = evaluate_f(self.problem_id, y, self.n_obj)
        for objective_index in range(1, self.n_obj + 1):
            alpha[objective_index - 1] = (objective_index**2.0) * float(alpha[objective_index - 1])
        return alpha

    def _beta(self, y: FloatArray, pareto_set_dimension: int) -> FloatArray:
        beta: FloatArray = _zeros(self.n_obj)
        if pareto_set_dimension == self.n_var:
            return beta

        g_values = evaluate_g(self._g_function_id(), y, pareto_set_dimension, self.n_var)
        z_values = y[pareto_set_dimension:] - g_values

        for idx in range(z_values.shape[0]):
            if abs(float(z_values[idx])) < _DOUBLE_MIN_VALUE:
                z_values[idx] = 0.0

        if self.bias:
            w_values = np.empty_like(z_values)
            for idx in range(z_values.shape[0]):
                w_values[idx] = zbias(float(z_values[idx]))
        else:
            w_values = z_values

        w_size = self.n_var - pareto_set_dimension
        for objective_index in range(1, self.n_obj + 1):
            j_values = zcat_get_j(objective_index, self.n_obj, w_values, w_size)
            z_score = zcat_evaluate_z(j_values, objective_index, self.imbalance, self.level)
            beta[objective_index - 1] = (objective_index**2.0) * z_score

        return beta

    def _evaluate_row(self, x_row: FloatArray) -> FloatArray:
        y = (x_row - self.xl) / (self.xu - self.xl)
        pareto_set_dimension = self._pareto_set_dimension(float(y[0]))
        alpha = self._alpha(y)
        beta = self._beta(y, pareto_set_dimension)
        result: FloatArray = _zeros(self.n_obj)
        for objective_index in range(self.n_obj):
            result[objective_index] = float(alpha[objective_index] + beta[objective_index])
        return result

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        x: FloatMatrix = np.asarray(X, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_var:
            raise ValueError(f"Expected decision matrix with shape (N, {self.n_var}).")

        f: FloatMatrix = np.empty((x.shape[0], self.n_obj), dtype=float)
        for row_index in range(x.shape[0]):
            f[row_index, :] = self._evaluate_row(x[row_index, :])

        if "F" in out and out["F"] is not None:
            out["F"][:] = f
        else:
            out["F"] = f


__all__ = ["ZCATProblem"]
