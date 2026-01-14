# Constraints and autodiff

Constraint handling strategies
------------------------------

Available strategies (see `vamos.foundation.constraints`):
- `feasibility_first`: feasible dominates infeasible; aggregates objectives (sum/max/none).
- `penalty_cv`: adds lambda * constraint violation to aggregated objectives.
- `cv_as_objective`: ranks by violation with small objective tie-break.
- `epsilon`: feasibility first with epsilon tolerance.

Constraint DSL
--------------

Build constraints symbolically, then evaluate on populations.

```python
import numpy as np
from vamos.foundation.constraints.dsl import constraint_model, build_constraint_evaluator

# Example: x0 + x1 <= 1 and x0 >= 0
with constraint_model(n_vars=2) as cm:
    x0, x1 = cm.vars("x0", "x1")
    cm.add(x0 + x1 <= 1.0)
    cm.add(x0 >= 0.0)

eval_constraints = build_constraint_evaluator(cm)
X = np.array([[0.2, 0.3], [0.9, 0.4]])
G = eval_constraints(X)  # shape (n_points, n_constraints), <=0 is satisfied
```

Notes:
- Constants in expressions must be scalar numbers (int/float/numpy scalar or 0-d array).
- Vector constants (lists/tuples/ndarrays with shape (n,)) are not supported; expand them into separate constraints.

Example (vector constants are not supported):

```python
# Not supported:
# cm.add(x0 <= np.array([1.0, 2.0]))

# Supported:
cm.add(x0 <= 1.0)
cm.add(x1 <= 2.0)
```

Autodiff (JAX)
--------------

Install `pip install -e ".[autodiff]"` to enable JAX-backed functions.

```python
from vamos.foundation.constraints.autodiff import build_jax_constraint_functions

fun, jac = build_jax_constraint_functions(cm)
vals = fun(X)        # same shape as G above
jacobian = jac(X)    # per-point Jacobians
```

Using constraints in algorithms
-------------------------------

- Provide problems that fill `out["G"]` in `evaluate` (shape n_points x n_constraints, <=0 satisfied).
- Algorithms like NSGA-II, MOEA/D, SMS-EMOA, SPEA2 honor constraints via the selected strategy inside kernels.
- Hypervolume early-stop and metrics continue to operate on objective values; feasibility affects selection and ranking.
