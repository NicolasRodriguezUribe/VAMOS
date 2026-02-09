"""VAMOS Studio -- Problem Builder backend helpers.

Pure-logic functions extracted from ``problem_builder.py`` to keep each
module under the LOC budget.  No Streamlit imports here.
"""

from __future__ import annotations

import textwrap
import time
from typing import Any

import numpy as np

_DEFAULT_TEMPLATE = "ZDT1-like (convex)"


def example_objectives() -> dict[str, dict[str, str]]:
    """Return example objective templates (lazy to avoid import-time side effects)."""
    return {
        **_math_templates(),
        **_domain_templates(),
        "Blank (write your own)": {
            "code": textwrap.dedent("""\
                f0 = x[0]
                f1 = x[1]
                return [f0, f1]"""),
            "n_var": "2",
            "n_obj": "2",
            "category": "blank",
        },
    }


def _math_templates() -> dict[str, dict[str, str]]:
    """Classic mathematical benchmark templates."""
    return {
        "ZDT1-like (convex)": {
            "code": textwrap.dedent("""\
                f0 = x[0]
                g  = 1.0 + 9.0 * sum(x[1:]) / (len(x) - 1)
                f1 = g * (1.0 - (x[0] / g) ** 0.5)
                return [f0, f1]"""),
            "n_var": "5",
            "n_obj": "2",
            "category": "math",
        },
        "Schaffer N.1 (concave)": {
            "code": textwrap.dedent("""\
                f0 = x[0] ** 2
                f1 = (x[0] - 2) ** 2
                return [f0, f1]"""),
            "n_var": "1",
            "n_obj": "2",
            "category": "math",
        },
        "Fonseca-Fleming": {
            "code": textwrap.dedent("""\
                import math
                n = len(x)
                s1 = sum((xi - 1.0 / n ** 0.5) ** 2 for xi in x)
                s2 = sum((xi + 1.0 / n ** 0.5) ** 2 for xi in x)
                f0 = 1.0 - math.exp(-s1)
                f1 = 1.0 - math.exp(-s2)
                return [f0, f1]"""),
            "n_var": "3",
            "n_obj": "2",
            "category": "math",
        },
        "Tri-objective (DTLZ1-like)": {
            "code": textwrap.dedent("""\
                g = 1.0 + sum(((xi - 0.5) ** 2) for xi in x[2:])
                f0 = 0.5 * x[0] * x[1] * (1 + g)
                f1 = 0.5 * x[0] * (1 - x[1]) * (1 + g)
                f2 = 0.5 * (1 - x[0]) * (1 + g)
                return [f0, f1, f2]"""),
            "n_var": "5",
            "n_obj": "3",
            "category": "math",
        },
    }


def _domain_templates() -> dict[str, dict[str, str]]:
    """Real-world domain-specific templates."""
    return {
        "Engineering: beam design (cost vs deflection)": {
            "code": textwrap.dedent("""\
                # x[0]=width, x[1]=height
                area = x[0] * x[1]
                cost = 2.0 * x[0] + 3.0 * x[1]  # material cost
                deflection = 1000.0 / (x[0] * x[1] ** 3 + 1e-6)  # stiffness
                return [cost, deflection]"""),
            "n_var": "2",
            "n_obj": "2",
            "bounds": "1.0, 10.0",
            "category": "engineering",
            "constraint_code": textwrap.dedent("""\
                # Stress must not exceed limit: stress <= 100
                stress = 600.0 / (x[0] * x[1] ** 2 + 1e-6)
                return [stress - 100.0]"""),
            "n_constraints": "1",
        },
        "ML: accuracy vs model size": {
            "code": textwrap.dedent("""\
                import math
                # x[0]=layers, x[1]=width_factor, x[2]=dropout
                params = x[0] * (x[1] ** 2) * 1000  # proxy for param count
                acc_proxy = 1.0 - math.exp(-0.5 * x[0] * x[1]) * (1.0 + 0.3 * x[2])
                neg_accuracy = -acc_proxy  # minimize negative accuracy
                model_size = params / 1e6  # in millions
                return [neg_accuracy, model_size]"""),
            "n_var": "3",
            "n_obj": "2",
            "bounds": "1.0, 10.0\n1.0, 8.0\n0.0, 0.5",
            "category": "ml",
        },
        "Scheduling: makespan vs tardiness": {
            "code": textwrap.dedent("""\
                # x[i] = priority weight for job i (continuous relaxation)
                n = len(x)
                # simulate: higher priority -> earlier start
                order = sorted(range(n), key=lambda i: -x[i])
                durations = [2 + i % 3 for i in range(n)]
                deadlines = [3 + 2 * i for i in range(n)]
                t = 0.0
                total_tardiness = 0.0
                for j in order:
                    t += durations[j]
                    total_tardiness += max(0.0, t - deadlines[j])
                makespan = t
                return [makespan, total_tardiness]"""),
            "n_var": "5",
            "n_obj": "2",
            "bounds": "0.0, 1.0",
            "category": "scheduling",
        },
    }


def compile_constraint_function(code: str) -> Any:
    """Compile user constraint code into ``g(x) -> list[float]``.

    Constraint values follow the convention: **g(x) <= 0 is feasible**.
    """
    import math as _math

    header = "import math\nimport numpy as np\n"
    full_source = header + "def _user_constraint(x):\n" + textwrap.indent(code, "    ") + "\n"
    local_ns: dict[str, Any] = {}
    exec(  # noqa: S102
        compile(full_source, "<constraint-builder>", "exec"),
        {"math": _math, "np": np, "__builtins__": __builtins__},
        local_ns,
    )
    fn = local_ns.get("_user_constraint")
    if fn is None:
        raise RuntimeError("Could not compile constraint function.")
    return fn


def compile_objective_function(code: str) -> Any:
    """Compile user code into a callable ``fn(x) -> list[float]``."""
    import math as _math

    header = "import math\nimport numpy as np\n"
    full_source = header + "def _user_fn(x):\n" + textwrap.indent(code, "    ") + "\n"
    local_ns: dict[str, Any] = {}
    exec(  # noqa: S102
        compile(full_source, "<problem-builder>", "exec"),
        {"math": _math, "np": np, "__builtins__": __builtins__},
        local_ns,
    )
    fn = local_ns.get("_user_fn")
    if fn is None:
        raise RuntimeError("Could not compile objective function.")
    return fn


def run_preview_optimization(
    fn: Any,
    *,
    n_var: int,
    n_obj: int,
    bounds: list[tuple[float, float]],
    algorithm: str,
    budget: int,
    pop_size: int,
    seed: int,
    constraints: Any = None,
    n_constraints: int = 0,
) -> dict[str, Any]:
    """Run a quick optimization and return ``{"F": ..., "X": ..., "elapsed_ms": ...}``."""
    from vamos.foundation.problem.builder import make_problem
    from vamos.ux.studio.services import _build_algorithm_config, _run_algorithm

    kw: dict[str, Any] = {}
    if constraints is not None and n_constraints > 0:
        kw["constraints"] = constraints
        kw["n_constraints"] = n_constraints
    problem = make_problem(fn, n_var=n_var, n_obj=n_obj, bounds=bounds, name="studio_preview", **kw)
    algo_cfg = _build_algorithm_config(algorithm, pop_size=pop_size, n_var=n_var, n_obj=n_obj)
    t0 = time.perf_counter()
    result = _run_algorithm(
        problem,
        algorithm=algorithm,
        algorithm_config=algo_cfg,
        termination=("max_evaluations", budget),
        seed=seed,
        engine="numpy",
    )
    elapsed = (time.perf_counter() - t0) * 1000.0
    F = result.get("F")
    X = result.get("X")
    if F is None:
        raise RuntimeError("Preview optimization returned no objectives.")
    return {
        "F": np.asarray(F),
        "X": np.asarray(X) if X is not None else None,
        "elapsed_ms": elapsed,
    }


def parse_bounds_text(text: str, n_var: int) -> list[tuple[float, float]] | str:
    """Parse a bounds textarea into a list of (lo, hi) tuples.

    Returns the list on success, or an error string on failure.
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) == 1:
        parts = lines[0].replace("(", "").replace(")", "").split(",")
        if len(parts) != 2:
            return "Enter two comma-separated numbers, e.g.  0.0, 1.0"
        try:
            lo, hi = float(parts[0].strip()), float(parts[1].strip())
        except ValueError:
            return "Bounds must be numbers."
        if lo > hi:
            return f"Lower bound ({lo}) must be <= upper bound ({hi})."
        return [(lo, hi)] * n_var

    if len(lines) != n_var:
        return f"Expected 1 line (same for all) or {n_var} lines (one per variable), got {len(lines)}."
    result: list[tuple[float, float]] = []
    for i, line in enumerate(lines):
        parts = line.replace("(", "").replace(")", "").split(",")
        if len(parts) != 2:
            return f"Line {i + 1}: expected two comma-separated numbers."
        try:
            lo, hi = float(parts[0].strip()), float(parts[1].strip())
        except ValueError:
            return f"Line {i + 1}: bounds must be numbers."
        if lo > hi:
            return f"Line {i + 1}: lower ({lo}) > upper ({hi})."
        result.append((lo, hi))
    return result


def generate_script(
    code: str,
    *,
    name: str,
    n_var: int,
    n_obj: int,
    bounds: list[tuple[float, float]],
    algorithm: str,
    budget: int,
    constraint_code: str = "",
    n_constraints: int = 0,
) -> str:
    """Generate a standalone Python script from the builder state."""
    func_name = "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_") or "custom"
    while "__" in func_name:
        func_name = func_name.replace("__", "_")

    has_constraints = bool(constraint_code.strip()) and n_constraints > 0

    parts: list[str] = [
        '"""',
        f"Custom problem: {name}",
        "",
        "Generated by VAMOS Studio -- Problem Builder.",
        '"""',
        "",
        "from vamos import make_problem, optimize",
        "",
        "",
        f"def {func_name}(x):",
        f'    """Evaluate one solution (x has length {n_var}). Returns {n_obj} objectives."""',
    ]
    for ln in code.splitlines():
        parts.append(f"    {ln}")

    if has_constraints:
        parts += [
            "",
            "",
            f"def {func_name}_constraints(x):",
            '    """Constraint function. Values <= 0 are feasible."""',
        ]
        for ln in constraint_code.splitlines():
            parts.append(f"    {ln}")

    parts += ["", "", "problem = make_problem(", f"    {func_name},"]
    parts.append(f"    n_var={n_var},")
    parts.append(f"    n_obj={n_obj},")
    parts.append(f"    bounds={bounds!r},")
    parts.append(f'    name="{name}",')
    if has_constraints:
        parts.append(f"    constraints={func_name}_constraints,")
        parts.append(f"    n_constraints={n_constraints},")
    parts.append(")")

    parts += [
        "",
        'if __name__ == "__main__":',
        "    result = optimize(",
        "        problem,",
        f'        algorithm="{algorithm}",',
        f"        max_evaluations={budget},",
        "        seed=42,",
        "        verbose=True,",
        "    )",
        "    F = result.F",
        '    print(f"\\nFound {len(F)} non-dominated solutions")',
    ]
    for i in range(n_obj):
        parts.append(f'    print(f"  f{i}: [{{F[:, {i}].min():.4f}}, {{F[:, {i}].max():.4f}}]")')
    parts.append("")

    return "\n".join(parts)


__all__ = [
    "compile_constraint_function",
    "compile_objective_function",
    "example_objectives",
    "generate_script",
    "parse_bounds_text",
    "run_preview_optimization",
]
