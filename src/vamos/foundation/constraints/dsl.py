from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Literal, Sequence

import numpy as np


@dataclass
class Var:
    name: str
    index: int

    def __repr__(self) -> str:  # pragma: no cover - debug
        return f"Var({self.name}@{self.index})"


class Expr:
    def __init__(self, op: str, args: Sequence[object]) -> None:
        self.op = op
        self.args: list[object] = list(args)

    def _coerce(self, other: Expr | Var | float | int | np.ndarray) -> Expr:
        if isinstance(other, (int, float, np.ndarray)):
            return Expr("const", [float(other)])
        if isinstance(other, Var):
            return Expr("var", [other])
        if isinstance(other, Expr):
            return other
        raise TypeError(f"Unsupported operand type {type(other)}")

    def __add__(self, other: Expr | Var | float | int | np.ndarray) -> Expr:
        return Expr("add", [self, self._coerce(other)])

    def __radd__(self, other: Expr | Var | float | int | np.ndarray) -> Expr:
        return self._coerce(other).__add__(self)

    def __sub__(self, other: Expr | Var | float | int | np.ndarray) -> Expr:
        return Expr("sub", [self, self._coerce(other)])

    def __rsub__(self, other: Expr | Var | float | int | np.ndarray) -> Expr:
        return self._coerce(other).__sub__(self)

    def __mul__(self, other: Expr | Var | float | int | np.ndarray) -> Expr:
        return Expr("mul", [self, self._coerce(other)])

    def __rmul__(self, other: Expr | Var | float | int | np.ndarray) -> Expr:
        return self._coerce(other).__mul__(self)

    def __truediv__(self, other: Expr | Var | float | int | np.ndarray) -> Expr:
        return Expr("div", [self, self._coerce(other)])

    def __rtruediv__(self, other: Expr | Var | float | int | np.ndarray) -> Expr:
        return self._coerce(other).__div__(self)

    def __pow__(self, power: Expr | Var | float | int | np.ndarray, modulo: object | None = None) -> Expr:
        return Expr("pow", [self, self._coerce(power)])

    def __le__(self, other: Expr | Var | float | int | np.ndarray) -> Constraint:
        return Constraint(lhs=self, rhs=self._coerce(other), sense="<=")

    def __ge__(self, other: Expr | Var | float | int | np.ndarray) -> Constraint:
        return Constraint(lhs=self, rhs=self._coerce(other), sense=">=")

    def __eq__(self, other: Expr | Var | float | int | np.ndarray) -> Constraint:
        return Constraint(lhs=self, rhs=self._coerce(other), sense="==")  # type: ignore[override]


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
        val = float(expr.args[0])
        return np.full(X.shape[0], val, dtype=float)
    if op == "var":
        var = expr.args[0]
        assert isinstance(var, Var)
        return X[:, var.index]
    a = _eval_expr(expr.args[0], X)
    b = _eval_expr(expr.args[1], X)
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "mul":
        return a * b
    if op == "div":
        return a / b
    if op == "pow":
        return np.power(a, b)
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
