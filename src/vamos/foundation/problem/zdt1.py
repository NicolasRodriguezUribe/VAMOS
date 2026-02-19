# problem/zdt1.py
import numpy as np
from vamos.foundation.problem.base import Problem



class ZDT1Problem(Problem):
    def __init__(self, n_var: int) -> None:
        self.n_var = n_var
        self.n_obj = 2
        # Bounds (identical for all decision variables in this problem)
        self.xl = 0.0
        self.xu = 1.0

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        """
        X: array shape (N, n_var)
        out["F"]: array shape (N, 2)
        """
        f1 = X[:, 0]
        g = 1.0 + 9.0 * np.mean(X[:, 1:], axis=1)
        f2 = g * (1.0 - np.sqrt(f1 / g))

        F = out["F"]
        F[:, 0] = f1
        F[:, 1] = f2
