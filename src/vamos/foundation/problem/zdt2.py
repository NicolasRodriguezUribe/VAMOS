import numpy as np


class ZDT2Problem:
    """
    Classic bi-objective benchmark with a concave Pareto front.
    Shares structure with ZDT1 but uses a quadratic term in the second objective.
    """

    def __init__(self, n_var: int):
        self.n_var = int(n_var)
        self.n_obj = 2
        self.xl = 0.0
        self.xu = 1.0

    def evaluate(self, X: np.ndarray, out: dict) -> None:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_var:
            raise ValueError(f"Expected input shape (N, {self.n_var}), got {X.shape}.")
        f1 = X[:, 0]
        g = 1.0 + 9.0 * np.mean(X[:, 1:], axis=1)
        f2 = g * (1.0 - (f1 / g) ** 2)
        F = out["F"]
        F[:, 0] = f1
        F[:, 1] = f2
