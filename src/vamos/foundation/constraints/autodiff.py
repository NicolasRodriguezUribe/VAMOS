from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np

from vamos.foundation.constraints.dsl import Constraint, ConstraintModel, Expr, Var


def _import_jax() -> tuple[Any, Any]:
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("JAX is required for autodiff constraints. Install with the 'autodiff' extra.") from exc
    return jax, jnp


def _expr_to_jax(expr: Expr, x: Any, jnp: Any) -> Any:
    op = expr.op
    if op == "const":
        return jnp.array(expr.args[0])
    if op == "var":
        var = cast(Var, expr.args[0])
        return x[var.index]
    a = _expr_to_jax(cast(Expr, expr.args[0]), x, jnp)
    b = _expr_to_jax(cast(Expr, expr.args[1]), x, jnp)
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "mul":
        return a * b
    if op == "div":
        return a / b
    if op == "pow":
        return a**b
    raise ValueError(f"Unsupported op {op}")


def _constraint_to_jax(c: Constraint, x: Any, jnp: Any) -> Any:
    lhs = _expr_to_jax(c.lhs, x, jnp)
    rhs = _expr_to_jax(c.rhs, x, jnp)
    if c.sense == "<=":
        return jnp.maximum(lhs - rhs, 0.0)
    if c.sense == ">=":
        return jnp.maximum(rhs - lhs, 0.0)
    return jnp.abs(lhs - rhs)


def build_jax_constraint_functions(
    cm: ConstraintModel,
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """
    Returns constraint_fun(x) and constraint_jac(x) for single x (1D array).
    Batch via jax.vmap externally if needed.
    """
    jax, jnp = _import_jax()
    constraints = list(cm.constraints)

    def single(x: Any) -> Any:
        return jnp.stack([_constraint_to_jax(c, x, jnp) for c in constraints])

    jac_single = jax.jacrev(single)
    batched_fun = jax.vmap(single)
    batched_jac = jax.vmap(jac_single)

    def fun(X: np.ndarray) -> np.ndarray:
        return np.array(batched_fun(jnp.asarray(X)))

    def jac(X: np.ndarray) -> np.ndarray:
        return np.array(batched_jac(jnp.asarray(X)))

    return fun, jac


__all__ = ["build_jax_constraint_functions"]
