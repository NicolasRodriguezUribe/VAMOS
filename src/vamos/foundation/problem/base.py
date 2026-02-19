"""
Base class for class-based custom optimization problems.
"""

from __future__ import annotations

import numpy as np


class Problem:
    """Base class for class-based custom optimization problems.

    Subclass this when your problem needs state — a dataset, distance matrix,
    simulator, or any data set up in ``__init__``.

    **Required:** set ``n_var``, ``n_obj``, ``xl``, ``xu`` in ``__init__``.
    **Optional:** override ``encoding`` and ``n_constraints`` as class-level
    attributes (not in ``__init__``).

    Example — unconstrained::

        import numpy as np
        from vamos import Problem, optimize

        class MyProblem(Problem):
            def __init__(self):
                self.n_var = 3
                self.n_obj = 2
                self.xl = np.zeros(3)
                self.xu = np.ones(3)

            def objectives(self, X: np.ndarray) -> np.ndarray:
                # X: (N, n_var) batch of candidate solutions
                f1 = np.sum(X ** 2, axis=1)
                f2 = np.sum((X - 1) ** 2, axis=1)
                return np.column_stack([f1, f2])

        result = optimize(MyProblem(), algorithm="nsgaii", max_evaluations=5000)

    Example — constrained::

        class MyConstrainedProblem(Problem):
            n_constraints = 1          # declare at class level

            def __init__(self):
                self.n_var = 3
                self.n_obj = 2
                self.xl = np.zeros(3)
                self.xu = np.ones(3)

            def objectives(self, X):
                f1 = np.sum(X ** 2, axis=1)
                f2 = np.sum((X - 1) ** 2, axis=1)
                return np.column_stack([f1, f2])

            def constraints(self, X):
                # Sign convention: g(x) <= 0 means feasible.
                g = np.sum(X, axis=1) - 2.0   # sum(x) <= 2
                return g.reshape(-1, 1)
    """

    # ------------------------------------------------------------------
    # Class-level defaults — override at class body level, not in __init__
    # ------------------------------------------------------------------

    encoding: str = "real"
    """Variable encoding.  Supported values: ``"real"``, ``"integer"``,
    ``"binary"``, ``"permutation"``, ``"mixed"``.  Default: ``"real"``."""

    n_constraints: int = 0
    """Number of inequality constraints.  Default: ``0`` (unconstrained)."""

    # ------------------------------------------------------------------
    # Engine compatibility
    # ------------------------------------------------------------------

    @property
    def n_constr(self) -> int:
        """Alias for :attr:`n_constraints`, used by evaluation backends."""
        return self.n_constraints

    # ------------------------------------------------------------------
    # User-overridable interface
    # ------------------------------------------------------------------

    def objectives(self, X: np.ndarray) -> np.ndarray:
        """Compute objective values for a batch of solutions.

        Override this method in your subclass.

        Args:
            X: Decision matrix of shape ``(N, n_var)`` where each row is a
               candidate solution.

        Returns:
            Array of shape ``(N, n_obj)`` with objective values to
            **minimize**.  A single-objective problem may return a 1-D array
            of length ``N``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement objectives(self, X)."
        )

    def constraints(self, X: np.ndarray) -> np.ndarray | None:
        """Compute constraint violations for a batch of solutions.

        Override this method when your problem has inequality constraints.

        Args:
            X: Decision matrix of shape ``(N, n_var)``.

        Returns:
            Array of shape ``(N, n_constraints)`` where **negative values
            indicate feasibility** (g(x) ≤ 0 is satisfied).  Positive values
            indicate constraint violation.  Return ``None`` if there are no
            constraints (the default).
        """
        return None

    # ------------------------------------------------------------------
    # Framework entry point — do not override
    # ------------------------------------------------------------------

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        """Framework evaluation entry point.  Override :meth:`objectives`
        (and optionally :meth:`constraints`) instead of this method."""
        X = np.asarray(X, dtype=float)

        # --- objectives ---
        F_computed = np.asarray(self.objectives(X), dtype=float)
        if F_computed.ndim == 1:
            F_computed = F_computed.reshape(-1, self.n_obj)
        F_buf = out.get("F")
        if F_buf is not None and F_buf.shape == F_computed.shape:
            F_buf[:] = F_computed
        else:
            out["F"] = F_computed

        # --- constraints ---
        if self.n_constraints > 0:
            G_computed = self.constraints(X)
            if G_computed is not None:
                G_computed = np.asarray(G_computed, dtype=float)
                if G_computed.ndim == 1:
                    G_computed = G_computed.reshape(-1, self.n_constraints)
                G_buf = out.get("G")
                if G_buf is not None and G_buf.shape == G_computed.shape:
                    G_buf[:] = G_computed
                else:
                    out["G"] = G_computed


__all__ = ["Problem"]
