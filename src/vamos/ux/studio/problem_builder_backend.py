"""VAMOS Studio -- Problem Builder backend helpers.

Pure-logic functions extracted from ``problem_builder.py`` to keep each
module under the LOC budget.  No Streamlit imports here.
"""

from __future__ import annotations

import ast
import builtins as _py_builtins
import textwrap
import time
from typing import Any

import numpy as np

_DEFAULT_TEMPLATE = "ZDT1-like (convex)"
_MAX_USER_CODE_CHARS = 12000
_PREVIEW_TIMEOUT_SECONDS = 10.0
_DEFAULT_SANDBOX_PROFILE = "basic"
_ALLOWED_IMPORT_ROOTS = {"math", "numpy"}
_SANDBOX_PROFILES: dict[str, dict[str, int]] = {
    "none": {},
    "basic": {
        "memory_mb": 2048,
        "max_file_bytes": 8_000_000,
        "max_open_files": 128,
        "max_processes": 64,
    },
    "strict": {
        "memory_mb": 1024,
        "max_file_bytes": 4_000_000,
        "max_open_files": 64,
        "max_processes": 32,
    },
}
_BLOCKED_NAME_REFERENCES = {
    "__import__",
    "open",
    "exec",
    "eval",
    "compile",
    "input",
    "help",
    "breakpoint",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
}
_SAFE_BUILTINS: dict[str, object] = {
    "abs": _py_builtins.abs,
    "all": _py_builtins.all,
    "any": _py_builtins.any,
    "bool": _py_builtins.bool,
    "dict": _py_builtins.dict,
    "enumerate": _py_builtins.enumerate,
    "float": _py_builtins.float,
    "int": _py_builtins.int,
    "isinstance": _py_builtins.isinstance,
    "len": _py_builtins.len,
    "list": _py_builtins.list,
    "max": _py_builtins.max,
    "min": _py_builtins.min,
    "pow": _py_builtins.pow,
    "range": _py_builtins.range,
    "round": _py_builtins.round,
    "set": _py_builtins.set,
    "sorted": _py_builtins.sorted,
    "str": _py_builtins.str,
    "sum": _py_builtins.sum,
    "tuple": _py_builtins.tuple,
    "zip": _py_builtins.zip,
    "Exception": _py_builtins.Exception,
    "ValueError": _py_builtins.ValueError,
    "TypeError": _py_builtins.TypeError,
    "RuntimeError": _py_builtins.RuntimeError,
    "IndexError": _py_builtins.IndexError,
    "KeyError": _py_builtins.KeyError,
}


def _safe_import(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:
    """Allow imports only from approved modules used in user formulas."""
    root = str(name).split(".", 1)[0]
    if level != 0 or root not in _ALLOWED_IMPORT_ROOTS:
        raise ImportError(f"Import '{name}' is not allowed in Studio preview code.")
    return _py_builtins.__import__(name, globals, locals, fromlist, level)


class _UserCodeSafetyVisitor(ast.NodeVisitor):
    """Reject clearly unsafe constructs in user-entered code."""

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            if root not in _ALLOWED_IMPORT_ROOTS:
                raise ValueError(
                    f"Import '{alias.name}' is not allowed. "
                    f"Allowed modules: {', '.join(sorted(_ALLOWED_IMPORT_ROOTS))}."
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is None:
            raise ValueError("Relative imports are not allowed in Studio preview code.")
        root = node.module.split(".", 1)[0]
        if node.level != 0 or root not in _ALLOWED_IMPORT_ROOTS:
            raise ValueError(
                f"Import from '{node.module}' is not allowed. "
                f"Allowed modules: {', '.join(sorted(_ALLOWED_IMPORT_ROOTS))}."
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _BLOCKED_NAME_REFERENCES:
            raise ValueError(f"'{node.id}' is not allowed in Studio preview code.")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            raise ValueError("Dunder attribute access is not allowed in Studio preview code.")
        self.generic_visit(node)


def _validate_user_code(code: str, *, section: str) -> None:
    if len(code) > _MAX_USER_CODE_CHARS:
        raise ValueError(f"{section} code is too long (>{_MAX_USER_CODE_CHARS} characters).")
    tree = ast.parse(code, mode="exec")
    _UserCodeSafetyVisitor().visit(tree)


def _normalize_sandbox_profile(profile: str) -> str:
    normalized = profile.strip().lower()
    if normalized not in _SANDBOX_PROFILES:
        options = ", ".join(sorted(_SANDBOX_PROFILES))
        raise ValueError(f"Unknown sandbox profile '{profile}'. Choose one of: {options}.")
    return normalized


def _try_set_rlimit(resource_mod: Any, limit_name: str, soft: int, hard: int) -> None:
    limit_const = getattr(resource_mod, limit_name, None)
    if limit_const is None:
        return
    try:
        resource_mod.setrlimit(limit_const, (int(soft), int(hard)))
    except (ValueError, OSError):
        # Best-effort only: OS and user privileges vary.
        return


def _apply_process_sandbox(*, profile: str, timeout_seconds: float) -> None:
    profile_name = _normalize_sandbox_profile(profile)
    if profile_name == "none":
        return
    try:
        import resource  # type: ignore[attr-defined]
    except Exception:
        # Resource limits are not available on all platforms (e.g., Windows).
        return

    settings = _SANDBOX_PROFILES[profile_name]
    cpu_soft = max(1, int(float(timeout_seconds)))
    cpu_hard = max(cpu_soft + 1, cpu_soft)
    _try_set_rlimit(resource, "RLIMIT_CPU", cpu_soft, cpu_hard)
    _try_set_rlimit(resource, "RLIMIT_CORE", 0, 0)

    mem_mb = settings.get("memory_mb")
    if mem_mb is not None:
        mem_bytes = int(mem_mb) * 1024 * 1024
        _try_set_rlimit(resource, "RLIMIT_AS", mem_bytes, mem_bytes)
        _try_set_rlimit(resource, "RLIMIT_DATA", mem_bytes, mem_bytes)

    max_file_bytes = settings.get("max_file_bytes")
    if max_file_bytes is not None:
        _try_set_rlimit(resource, "RLIMIT_FSIZE", int(max_file_bytes), int(max_file_bytes))

    max_open_files = settings.get("max_open_files")
    if max_open_files is not None:
        _try_set_rlimit(resource, "RLIMIT_NOFILE", int(max_open_files), int(max_open_files))

    max_processes = settings.get("max_processes")
    if max_processes is not None:
        _try_set_rlimit(resource, "RLIMIT_NPROC", int(max_processes), int(max_processes))


def _compile_user_function(code: str, *, func_name: str, source_tag: str) -> Any:
    import math

    _validate_user_code(code, section=func_name)
    source = f"def {func_name}(x):\n" + textwrap.indent(code, "    ") + "\n"
    local_ns: dict[str, Any] = {}
    safe_builtins = dict(_SAFE_BUILTINS)
    safe_builtins["__import__"] = _safe_import
    exec(  # noqa: S102
        compile(source, source_tag, "exec"),
        {"math": math, "np": np, "__builtins__": safe_builtins},
        local_ns,
    )
    fn = local_ns.get(func_name)
    if fn is None:
        raise RuntimeError(f"Could not compile {func_name}.")
    return fn


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
    return _compile_user_function(
        code,
        func_name="_user_constraint",
        source_tag="<constraint-builder>",
    )


def compile_objective_function(code: str) -> Any:
    """Compile user code into a callable ``fn(x) -> list[float]``."""
    return _compile_user_function(
        code,
        func_name="_user_fn",
        source_tag="<problem-builder>",
    )


def _run_preview_once(
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
    from vamos.foundation.problem.builder import make_problem
    from vamos.ux.studio.services import _build_algorithm_config, _run_algorithm

    kw: dict[str, Any] = {}
    if constraints is not None and n_constraints > 0:
        kw["constraints"] = constraints
        kw["n_constraints"] = n_constraints
    problem = make_problem(fn, n_var=n_var, n_obj=n_obj, bounds=bounds, encoding="real", name="studio_preview", **kw)
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


def _preview_worker(
    queue: Any,
    *,
    objective_code: str,
    constraint_code: str,
    n_var: int,
    n_obj: int,
    bounds: list[tuple[float, float]],
    algorithm: str,
    budget: int,
    pop_size: int,
    seed: int,
    n_constraints: int,
    timeout_seconds: float,
    sandbox_profile: str,
) -> None:
    try:
        _apply_process_sandbox(profile=sandbox_profile, timeout_seconds=timeout_seconds)
        fn = compile_objective_function(objective_code)
        constraints = None
        if constraint_code.strip() and n_constraints > 0:
            constraints = compile_constraint_function(constraint_code)
        payload = _run_preview_once(
            fn,
            n_var=n_var,
            n_obj=n_obj,
            bounds=bounds,
            algorithm=algorithm,
            budget=budget,
            pop_size=pop_size,
            seed=seed,
            constraints=constraints,
            n_constraints=n_constraints,
        )
        queue.put({"ok": True, "payload": payload})
    except Exception as exc:  # pragma: no cover - validated by parent behavior tests
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


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
    objective_code: str | None = None,
    constraint_code: str = "",
    timeout_seconds: float = _PREVIEW_TIMEOUT_SECONDS,
    sandbox_profile: str = _DEFAULT_SANDBOX_PROFILE,
) -> dict[str, Any]:
    """Run a quick optimization and return ``{"F": ..., "X": ..., "elapsed_ms": ...}``."""
    sandbox_profile = _normalize_sandbox_profile(sandbox_profile)
    if objective_code is None or timeout_seconds <= 0:
        return _run_preview_once(
            fn,
            n_var=n_var,
            n_obj=n_obj,
            bounds=bounds,
            algorithm=algorithm,
            budget=budget,
            pop_size=pop_size,
            seed=seed,
            constraints=constraints,
            n_constraints=n_constraints,
        )

    import multiprocessing as mp
    import queue as queue_mod

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_preview_worker,
        kwargs={
            "queue": result_queue,
            "objective_code": objective_code,
            "constraint_code": constraint_code,
            "n_var": n_var,
            "n_obj": n_obj,
            "bounds": bounds,
            "algorithm": algorithm,
            "budget": budget,
            "pop_size": pop_size,
            "seed": seed,
            "n_constraints": n_constraints,
            "timeout_seconds": timeout_seconds,
            "sandbox_profile": sandbox_profile,
        },
    )
    process.start()
    process.join(float(timeout_seconds))

    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        raise TimeoutError(f"Preview timed out after {timeout_seconds:.1f}s. Try simpler code or a smaller budget.")

    try:
        worker_result = result_queue.get(timeout=1.0)
    except queue_mod.Empty as exc:
        if process.exitcode not in (0, None):
            raise RuntimeError(f"Preview worker exited unexpectedly (exit code {process.exitcode}).") from exc
        raise RuntimeError("Preview worker finished without returning a result.") from exc
    finally:
        result_queue.close()
        result_queue.join_thread()

    if not isinstance(worker_result, dict):
        raise RuntimeError("Preview worker returned an invalid result payload.")
    if not worker_result.get("ok", False):
        raise RuntimeError(str(worker_result.get("error", "Unknown preview worker error.")))
    payload = worker_result.get("payload")
    if not isinstance(payload, dict):
        raise RuntimeError("Preview worker returned an invalid payload body.")
    payload["F"] = np.asarray(payload.get("F"))
    payload["X"] = np.asarray(payload["X"]) if payload.get("X") is not None else None
    return payload


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
    parts.append('    encoding="real",')
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
