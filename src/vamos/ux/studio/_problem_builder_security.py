from __future__ import annotations

import ast
import builtins as _py_builtins
import textwrap
from typing import Any

import numpy as np

_MAX_USER_CODE_CHARS = 12000
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


def safe_import(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:
    """Allow imports only from approved modules used in user formulas."""
    root = str(name).split(".", 1)[0]
    if level != 0 or root not in _ALLOWED_IMPORT_ROOTS:
        raise ImportError(f"Import '{name}' is not allowed in Studio preview code.")
    return _py_builtins.__import__(name, globals, locals, fromlist, level)


class UserCodeSafetyVisitor(ast.NodeVisitor):
    """Reject clearly unsafe constructs in user-entered code."""

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            if root not in _ALLOWED_IMPORT_ROOTS:
                raise ValueError(f"Import '{alias.name}' is not allowed. Allowed modules: {', '.join(sorted(_ALLOWED_IMPORT_ROOTS))}.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is None:
            raise ValueError("Relative imports are not allowed in Studio preview code.")
        root = node.module.split(".", 1)[0]
        if node.level != 0 or root not in _ALLOWED_IMPORT_ROOTS:
            raise ValueError(f"Import from '{node.module}' is not allowed. Allowed modules: {', '.join(sorted(_ALLOWED_IMPORT_ROOTS))}.")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _BLOCKED_NAME_REFERENCES:
            raise ValueError(f"'{node.id}' is not allowed in Studio preview code.")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            raise ValueError("Dunder attribute access is not allowed in Studio preview code.")
        self.generic_visit(node)


def validate_user_code(code: str, *, section: str) -> None:
    if len(code) > _MAX_USER_CODE_CHARS:
        raise ValueError(f"{section} code is too long (>{_MAX_USER_CODE_CHARS} characters).")
    tree = ast.parse(code, mode="exec")
    UserCodeSafetyVisitor().visit(tree)


def normalize_sandbox_profile(profile: str) -> str:
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
        return


def apply_process_sandbox(*, profile: str, timeout_seconds: float) -> None:
    profile_name = normalize_sandbox_profile(profile)
    if profile_name == "none":
        return
    try:
        import resource
    except Exception:
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


def compile_user_function(code: str, *, func_name: str, source_tag: str) -> Any:
    import math

    validate_user_code(code, section=func_name)
    source = f"def {func_name}(x):\n" + textwrap.indent(code, "    ") + "\n"
    local_ns: dict[str, Any] = {}
    safe_builtins = dict(_SAFE_BUILTINS)
    safe_builtins["__import__"] = safe_import
    exec(  # noqa: S102
        compile(source, source_tag, "exec"),
        {"math": math, "np": np, "__builtins__": safe_builtins},
        local_ns,
    )
    fn = local_ns.get(func_name)
    if fn is None:
        raise RuntimeError(f"Could not compile {func_name}.")
    return fn


def compile_constraint_function(code: str) -> Any:
    return compile_user_function(code, func_name="_user_constraint", source_tag="<constraint-builder>")


def compile_objective_function(code: str) -> Any:
    return compile_user_function(code, func_name="_user_fn", source_tag="<problem-builder>")


__all__ = [
    "apply_process_sandbox",
    "compile_constraint_function",
    "compile_objective_function",
    "normalize_sandbox_profile",
]
