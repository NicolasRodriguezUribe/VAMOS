from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

ProblemFactory = Callable[[int, int | None], object]
DefaultNVarFn = Callable[[int], int]


@dataclass(frozen=True)
class ProblemSpec:
    """Metadata and factory for a benchmark problem."""

    key: str
    label: str
    default_n_var: int
    default_n_obj: int
    allow_n_obj_override: bool
    factory: ProblemFactory
    default_n_var_fn: DefaultNVarFn | None = None
    description: str = ""
    encoding: str = "continuous"

    def resolve_dimensions(self, *, n_var: int | None, n_obj: int | None) -> tuple[int, int]:
        """
        Apply default dimensions and enforce override rules.
        """
        if self.allow_n_obj_override:
            actual_n_obj = n_obj if n_obj is not None else self.default_n_obj
            if actual_n_obj <= 0:
                raise ValueError("n_obj must be a positive integer.")
        else:
            actual_n_obj = self.default_n_obj
            if n_obj is not None and n_obj != actual_n_obj:
                raise ValueError(
                    f"Problem '{self.label}' has a fixed number of objectives ({self.default_n_obj}). --n-obj overrides are not supported."
                )

        if n_var is None:
            actual_n_var = self.default_n_var_fn(actual_n_obj) if self.default_n_var_fn is not None else self.default_n_var
        else:
            actual_n_var = n_var
        if actual_n_var <= 0:
            raise ValueError("n_var must be a positive integer.")

        return actual_n_var, actual_n_obj


__all__ = ["ProblemSpec", "ProblemFactory"]
