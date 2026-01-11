from __future__ import annotations

import numpy as np


class _CEC2009UFBase:
    """Base utilities shared by UF1-UF3 implementations."""

    def __init__(self, n_var: int = 30) -> None:
        if n_var < 3:
            raise ValueError("CEC2009 UF problems require at least 3 decision variables.")
        self.n_var = int(n_var)
        self.n_obj = 2
        self.encoding = "continuous"
        xl = np.full(self.n_var, -1.0, dtype=float)
        xu = np.ones(self.n_var, dtype=float)
        xl[0] = 0.0
        self.xl = xl
        self.xu = xu

    def _split_even_odd(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # j indices correspond to the literature's 1-based variables (j=2..n_var)
        j = np.arange(2, self.n_var + 1)
        even_mask = j % 2 == 0
        odd_mask = ~even_mask
        return j, even_mask, odd_mask


class CEC2009_UF1(_CEC2009UFBase):
    """UF1 from the CEC2009 competition."""

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_var:
            raise ValueError(f"Expected decision matrix of shape (N, {self.n_var}).")
        n = X.shape[0]
        F = out.get("F")
        if F is None or F.shape != (n, 2):
            F = np.empty((n, 2), dtype=float)
            out["F"] = F

        x0 = X[:, 0]
        rest = X[:, 1:]
        j, even_mask, odd_mask = self._split_even_odd()
        angles = 6.0 * np.pi * x0[:, None] + j * np.pi / self.n_var
        y = rest - np.sin(angles)
        y2 = y * y

        sum_odd = np.sum(y2[:, odd_mask], axis=1)
        sum_even = np.sum(y2[:, even_mask], axis=1)
        count_odd = max(1, int(odd_mask.sum()))
        count_even = max(1, int(even_mask.sum()))

        F[:, 0] = x0 + 2.0 * sum_odd / count_odd
        F[:, 1] = 1.0 - np.sqrt(x0) + 2.0 * sum_even / count_even


class CEC2009_UF2(_CEC2009UFBase):
    """UF2 from the CEC2009 competition."""

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_var:
            raise ValueError(f"Expected decision matrix of shape (N, {self.n_var}).")
        n = X.shape[0]
        F = out.get("F")
        if F is None or F.shape != (n, 2):
            F = np.empty((n, 2), dtype=float)
            out["F"] = F

        x0 = X[:, 0]
        rest = X[:, 1:]
        j, even_mask, odd_mask = self._split_even_odd()
        base = 0.3 * x0[:, None] * (x0[:, None] * np.cos(24.0 * np.pi * x0[:, None] + 4.0 * j * np.pi / self.n_var) + 2.0)
        angles = 6.0 * np.pi * x0[:, None] + j * np.pi / self.n_var
        sin_term = np.sin(angles)
        cos_term = np.cos(angles)

        y_even = rest[:, even_mask] - base[:, even_mask] * sin_term[:, even_mask]
        y_odd = rest[:, odd_mask] - base[:, odd_mask] * cos_term[:, odd_mask]

        sum_even = np.sum(y_even * y_even, axis=1)
        sum_odd = np.sum(y_odd * y_odd, axis=1)
        count_even = max(1, int(even_mask.sum()))
        count_odd = max(1, int(odd_mask.sum()))

        F[:, 0] = x0 + 2.0 * sum_odd / count_odd
        F[:, 1] = 1.0 - np.sqrt(x0) + 2.0 * sum_even / count_even


class CEC2009_UF3(_CEC2009UFBase):
    """UF3 from the CEC2009 competition."""

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_var:
            raise ValueError(f"Expected decision matrix of shape (N, {self.n_var}).")
        n = X.shape[0]
        F = out.get("F")
        if F is None or F.shape != (n, 2):
            F = np.empty((n, 2), dtype=float)
            out["F"] = F

        x0 = X[:, 0]
        rest = X[:, 1:]
        j, even_mask, odd_mask = self._split_even_odd()
        power = 0.5 * (1.0 + 3.0 * (j - 2.0) / (self.n_var - 2.0))
        y = rest - np.power(x0[:, None], power)
        p = np.cos(20.0 * y * np.pi / np.sqrt(j))

        y_even = y[:, even_mask]
        y_odd = y[:, odd_mask]
        p_even = p[:, even_mask]
        p_odd = p[:, odd_mask]

        sum_even = np.sum(y_even * y_even, axis=1)
        sum_odd = np.sum(y_odd * y_odd, axis=1)
        prod_even = np.prod(p_even, axis=1)
        prod_odd = np.prod(p_odd, axis=1)

        count_even = max(1, int(even_mask.sum()))
        count_odd = max(1, int(odd_mask.sum()))

        F[:, 0] = x0 + 2.0 * (4.0 * sum_odd - 2.0 * prod_odd + 2.0) / count_odd
        F[:, 1] = 1.0 - np.sqrt(x0) + 2.0 * (4.0 * sum_even - 2.0 * prod_even + 2.0) / count_even


class CEC2009_CF1(_CEC2009UFBase):
    """Constrained CF1 from the CEC2009 competition."""

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_var:
            raise ValueError(f"Expected decision matrix of shape (N, {self.n_var}).")
        n = X.shape[0]
        F = out.get("F")
        if F is None or F.shape != (n, 2):
            F = np.empty((n, 2), dtype=float)
            out["F"] = F

        x0 = X[:, 0]
        rest = X[:, 1:]
        j, even_mask, odd_mask = self._split_even_odd()
        power = 0.5 * (1.0 + 3.0 * (j - 2.0) / (self.n_var - 2.0))
        y = rest - np.power(x0[:, None], power)

        y_odd = y[:, odd_mask]
        y_even = y[:, even_mask]
        sum_odd = np.sum(y_odd * y_odd, axis=1)
        sum_even = np.sum(y_even * y_even, axis=1)
        count_odd = max(1, int(odd_mask.sum()))
        count_even = max(1, int(even_mask.sum()))

        f0 = x0 + 2.0 * sum_odd / count_odd
        f1 = 1.0 - x0 + 2.0 * sum_even / count_even
        N = 10.0
        a = 1.0
        constraint = f1 + f0 - a * np.abs(np.sin(N * np.pi * (f0 - f1 + 1.0))) - 1.0

        F[:, 0] = f0
        F[:, 1] = f1
        out["G"] = constraint[:, None]


__all__ = ["CEC2009_UF1", "CEC2009_UF2", "CEC2009_UF3", "CEC2009_CF1"]
