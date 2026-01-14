from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal
from collections.abc import Callable, Iterator, Sequence

import numpy as np


@dataclass
class Var:
    name: str
    index: int

    def __repr__(self) -> str:  # pragma: no cover - debug
        return f"Var({self.name}@{self.index})"


class Expr:
    def __init__(self, op: str, args: Sequence[Expr | Var | float]) -> None:
        self.op = op
        self.args: list[Expr | Var | float] = list(args)

    def _coerce(self, other: Expr | Var | float | int | np.number | np.ndarray) -> Expr:
        if isinstance(other, np.ndarray):
            if other.ndim == 0:
                return Expr("const", [float(other)])
            raise TypeError("Vector constants are not supported; use scalar values or expand into separate constraints.")
        if isinstance(other, (int, float, np.number)):
            return Expr("const", [float(other)])
        if isinstance(other, Var):
            return Expr("var", [other])
        if isinstance(other, Expr):
            return other
        raise TypeError(f"Unsupported operand type (scalar expected): {type(other)}")

    def __add__(self, other: Expr | Var | float | int | np.number | np.ndarray) -> Expr:
        return Expr("add", [self, self._coerce(other)])

    def __radd__(self, other: Expr | Var | float | int | np.number | np.ndarray) -> Expr:
        return self._coerce(other).__add__(self)

    def __sub__(self, other: Expr | Var | float | int | np.number | np.ndarray) -> Expr:
        return Expr("sub", [self, self._coerce(other)])

    def __rsub__(self, other: Expr | Var | float | int | np.number | np.ndarray) -> Expr:
        return self._coerce(other).__sub__(self)

    def __mul__(self, other: Expr | Var | float | int | np.number | np.ndarray) -> Expr:
        return Expr("mul", [self, self._coerce(other)])

    def __rmul__(self, other: Expr | Var | float | int | np.number | np.ndarray) -> Expr:
        return self._coerce(other).__mul__(self)

    def __truediv__(self, other: Expr | Var | float | int | np.number | np.ndarray) -> Expr:
        return Expr("div", [self, self._coerce(other)])

    def __rtruediv__(self, other: Expr | Var | float | int | np.number | np.ndarray) -> Expr:
        return Expr("div", [self._coerce(other), self])

    def __pow__(
        self,
        power: Expr | Var | float | int | np.number | np.ndarray,
        modulo: object | None = None,
    ) -> Expr:
        return Expr("pow", [self, self._coerce(power)])

    def __le__(self, other: Expr | Var | float | int | np.number | np.ndarray) -> Constraint:
        return Constraint(lhs=self, rhs=self._coerce(other), sense="<=")

    def __ge__(self, other: Expr | Var | float | int | np.number | np.ndarray) -> Constraint:
        return Constraint(lhs=self, rhs=self._coerce(other), sense=">=")

    def __eq__(self, other: Expr | Var | float | int | np.number | np.ndarray) -> Constraint:  # type: ignore[override]
        return Constraint(lhs=self, rhs=self._coerce(other), sense="==")


@dataclass
class Constraint:
    lhs: Expr
    rhs: Expr
    sense: Literal["<=", ">=", "=="]


class ConstraintModel:
    def __init__(self, n_vars: int) -> None:
        self.n_vars = n_vars
        self.vars_list: list[Var] = []
        self.constraints: list[Constraint] = []

    def vars(self, *names: str) -> tuple[Expr, ...]:
        start = len(self.vars_list)
        created: list[Expr] = []
        for i, name in enumerate(names):
            idx = start + i
            if idx >= self.n_vars:
                raise ValueError("Number of vars exceeds n_vars in model.")
            v = Var(name=name, index=idx)
            self.vars_list.append(v)
            created.append(Expr("var", [v]))
        return tuple(created)

    def add(self, constraint: Constraint) -> None:
        self.constraints.append(constraint)


@contextmanager
def constraint_model(n_vars: int) -> Iterator[ConstraintModel]:
    cm = ConstraintModel(n_vars=n_vars)
    yield cm


def _eval_expr(expr: Expr, X: np.ndarray) -> np.ndarray:
    op = expr.op
    if op == "const":
        const = expr.args[0]
        if not isinstance(const, (int, float)):
            raise TypeError("Const expressions must store numeric values.")
        val = float(const)
        return np.full(X.shape[0], val, dtype=float)
    if op == "var":
        var = expr.args[0]
        assert isinstance(var, Var)
        return X[:, var.index]
    lhs = expr.args[0]
    rhs = expr.args[1]
    if not isinstance(lhs, Expr) or not isinstance(rhs, Expr):
        raise TypeError("Binary expressions must have Expr operands.")
    a = _eval_expr(lhs, X)
    b = _eval_expr(rhs, X)
    if op == "add":
        return np.asarray(a + b, dtype=float)
    if op == "sub":
        return np.asarray(a - b, dtype=float)
    if op == "mul":
        return np.asarray(a * b, dtype=float)
    if op == "div":
        return np.asarray(a / b, dtype=float)
    if op == "pow":
        return np.asarray(np.power(a, b), dtype=float)
    raise ValueError(f"Unsupported op {op}")


def build_constraint_evaluator(cm: ConstraintModel) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build a vectorized evaluator returning violations shape (N, n_constraints).
    """
    constraints = list(cm.constraints)

    def eval_constraints(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        violations = np.zeros((X.shape[0], len(constraints)), dtype=float)
        for idx, c in enumerate(constraints):
            lhs = _eval_expr(c.lhs, X)
            rhs = _eval_expr(c.rhs, X)
            if c.sense == "<=":
                violations[:, idx] = np.maximum(lhs - rhs, 0.0)
            elif c.sense == ">=":
                violations[:, idx] = np.maximum(rhs - lhs, 0.0)
            else:  # equality
                violations[:, idx] = np.abs(lhs - rhs)
        return violations

    return eval_constraints


__all__ = [
    "Var",
    "Expr",
    "Constraint",
    "ConstraintModel",
    "constraint_model",
    "build_constraint_evaluator",
]
