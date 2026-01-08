import numpy as np


class DTLZBase:
    def __init__(self, n_var: int, n_obj: int):
        self.n_var = n_var
        self.n_obj = n_obj
        self.xl = 0.0
        self.xu = 1.0

    def _evaluate(self, X: np.ndarray, out: dict):
        raise NotImplementedError

    def evaluate(self, X: np.ndarray, out: dict):
        self._evaluate(X, out)


class DTLZ1Problem(DTLZBase):
    def __init__(self, n_var: int = 7, n_obj: int = 3):
        super().__init__(n_var, n_obj)

    def _evaluate(self, X: np.ndarray, out: dict):
        g = 100.0 * (
            self.n_var
            - self.n_obj
            + 1
            + np.sum(
                (X[:, self.n_obj - 1 :] - 0.5) ** 2 - np.cos(20 * np.pi * (X[:, self.n_obj - 1 :] - 0.5)),
                axis=1,
            )
        )
        F = np.ones((X.shape[0], self.n_obj))
        for i in range(self.n_obj):
            f = np.ones(X.shape[0])
            for j in range(self.n_obj - i - 1):
                f *= X[:, j]
            if i > 0:
                idx = self.n_obj - i - 1
                f *= 1.0 - X[:, idx]
            F[:, i] = f

        F_res = 0.5 * (1.0 + g[:, None]) * F
        if "F" in out and out["F"] is not None:
            out["F"][:] = F_res
        else:
            out["F"] = F_res


class DTLZ2Problem(DTLZBase):
    def __init__(self, n_var: int = 12, n_obj: int = 3):
        super().__init__(n_var, n_obj)

    def _evaluate(self, X: np.ndarray, out: dict):
        g = np.sum((X[:, self.n_obj - 1 :] - 0.5) ** 2, axis=1)
        F = np.ones((X.shape[0], self.n_obj))
        for i in range(self.n_obj):
            f = np.ones(X.shape[0])
            for j in range(self.n_obj - i - 1):
                f *= np.cos(X[:, j] * np.pi / 2.0)
            if i > 0:
                idx = self.n_obj - i - 1
                f *= np.sin(X[:, idx] * np.pi / 2.0)
            F[:, i] = f

        F_res = (1.0 + g[:, None]) * F
        if "F" in out and out["F"] is not None:
            out["F"][:] = F_res
        else:
            out["F"] = F_res


class DTLZ3Problem(DTLZBase):
    def __init__(self, n_var: int = 12, n_obj: int = 3):
        super().__init__(n_var, n_obj)

    def _evaluate(self, X: np.ndarray, out: dict):
        g = 100.0 * (
            self.n_var
            - self.n_obj
            + 1
            + np.sum(
                (X[:, self.n_obj - 1 :] - 0.5) ** 2 - np.cos(20 * np.pi * (X[:, self.n_obj - 1 :] - 0.5)),
                axis=1,
            )
        )
        F = np.ones((X.shape[0], self.n_obj))
        for i in range(self.n_obj):
            f = np.ones(X.shape[0])
            for j in range(self.n_obj - i - 1):
                f *= np.cos(X[:, j] * np.pi / 2.0)
            if i > 0:
                idx = self.n_obj - i - 1
                f *= np.sin(X[:, idx] * np.pi / 2.0)
            F[:, i] = f

        F_res = (1.0 + g[:, None]) * F
        if "F" in out and out["F"] is not None:
            out["F"][:] = F_res
        else:
            out["F"] = F_res


class DTLZ4Problem(DTLZBase):
    def __init__(self, n_var: int = 12, n_obj: int = 3, alpha: float = 100.0):
        super().__init__(n_var, n_obj)
        self.alpha = alpha

    def _evaluate(self, X: np.ndarray, out: dict):
        g = np.sum((X[:, self.n_obj - 1 :] - 0.5) ** 2, axis=1)
        alpha = self.alpha
        F = np.ones((X.shape[0], self.n_obj))
        for i in range(self.n_obj):
            f = np.ones(X.shape[0])
            for j in range(self.n_obj - i - 1):
                f *= np.cos((X[:, j] ** alpha) * np.pi / 2.0)
            if i > 0:
                idx = self.n_obj - i - 1
                f *= np.sin((X[:, idx] ** alpha) * np.pi / 2.0)
            F[:, i] = f

        F_res = (1.0 + g[:, None]) * F
        if "F" in out and out["F"] is not None:
            out["F"][:] = F_res
        else:
            out["F"] = F_res


class DTLZ7Problem(DTLZBase):
    """DTLZ7: Problem with disconnected Pareto-optimal regions."""

    def __init__(self, n_var: int = 22, n_obj: int = 3):
        super().__init__(n_var, n_obj)

    def _evaluate(self, X: np.ndarray, out: dict):
        # g function uses the last k variables
        g = 1.0 + (9.0 / (self.n_var - self.n_obj + 1)) * np.sum(X[:, self.n_obj - 1 :], axis=1)

        # First M-1 objectives are just x_i
        F = np.zeros((X.shape[0], self.n_obj))
        for i in range(self.n_obj - 1):
            F[:, i] = X[:, i]

        # Last objective: h function
        h = self.n_obj - np.sum(
            (F[:, : self.n_obj - 1] / (1.0 + g[:, None])) * (1.0 + np.sin(3.0 * np.pi * F[:, : self.n_obj - 1])),
            axis=1,
        )
        F[:, self.n_obj - 1] = (1.0 + g) * h

        if "F" in out and out["F"] is not None:
            out["F"][:] = F
        else:
            out["F"] = F
