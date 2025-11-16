"""
Wrappers for the full WFG problem suite (WFG1-WFG9).
These implementations rely on pymoo's reference formulations. Install pymoo if you intend to
instantiate any of these classes.
"""

import numpy as np

try:
    from pymoo.problems import get_problem as _get_wfg_problem
except ImportError:  # pragma: no cover
    _get_wfg_problem = None


class _BaseWFG:
    def __init__(self, name: str, n_var: int, n_obj: int, k: int | None = None, l: int | None = None):
        if _get_wfg_problem is None:
            raise ImportError(
                "WFG problem classes require the 'pymoo' dependency. Install it with "
                "`pip install pymoo` to enable these benchmarks."
            )
        if k is None:
            k = 2 * (n_obj - 1)
        if k % (n_obj - 1) != 0:
            raise ValueError("k must be divisible by (n_obj - 1)")
        if l is None:
            l = n_var - k
        if k + l != n_var:
            raise ValueError("n_var must equal k + l in WFG problems.")

        self._problem = _get_wfg_problem(name, n_var=n_var, n_obj=n_obj, k=k, l=l)
        self.n_var = self._problem.n_var
        self.n_obj = self._problem.n_obj
        self.k = k
        self.l = l
        # pymoo WFG problems operate in [0,1]
        self.xl = 0.0
        self.xu = 1.0

    def evaluate(self, X: np.ndarray, out: dict):
        F = np.empty((X.shape[0], self.n_obj))
        self._problem._evaluate(X, out={"F": F})
        out["F"] = F


class WFG1Problem(_BaseWFG):
    def __init__(self, n_var: int = 24, n_obj: int = 3, k: int | None = None, l: int | None = None):
        super().__init__("wfg1", n_var, n_obj, k, l)


class WFG2Problem(_BaseWFG):
    def __init__(self, n_var: int = 24, n_obj: int = 3, k: int | None = None, l: int | None = None):
        super().__init__("wfg2", n_var, n_obj, k, l)


class WFG3Problem(_BaseWFG):
    def __init__(self, n_var: int = 24, n_obj: int = 3, k: int | None = None, l: int | None = None):
        super().__init__("wfg3", n_var, n_obj, k, l)


class WFG4Problem(_BaseWFG):
    def __init__(self, n_var: int = 24, n_obj: int = 3, k: int | None = None, l: int | None = None):
        super().__init__("wfg4", n_var, n_obj, k, l)


class WFG5Problem(_BaseWFG):
    def __init__(self, n_var: int = 24, n_obj: int = 3, k: int | None = None, l: int | None = None):
        super().__init__("wfg5", n_var, n_obj, k, l)


class WFG6Problem(_BaseWFG):
    def __init__(self, n_var: int = 24, n_obj: int = 3, k: int | None = None, l: int | None = None):
        super().__init__("wfg6", n_var, n_obj, k, l)


class WFG7Problem(_BaseWFG):
    def __init__(self, n_var: int = 24, n_obj: int = 3, k: int | None = None, l: int | None = None):
        super().__init__("wfg7", n_var, n_obj, k, l)


class WFG8Problem(_BaseWFG):
    def __init__(self, n_var: int = 24, n_obj: int = 3, k: int | None = None, l: int | None = None):
        super().__init__("wfg8", n_var, n_obj, k, l)


class WFG9Problem(_BaseWFG):
    def __init__(self, n_var: int = 24, n_obj: int = 3, k: int | None = None, l: int | None = None):
        super().__init__("wfg9", n_var, n_obj, k, l)
