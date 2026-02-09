"""Architecture gate: no bare ``except:`` or silently swallowed exceptions.

This test scans all Python files under ``src/vamos/`` and rejects:

1. **Bare ``except:``** -- always use a specific exception type.
2. **Silent swallow** -- ``except SomeError: pass`` with no logging,
   re-raise, or any other statement.  A bare ``pass`` hides failures
   and makes debugging nearly impossible.

Allowed patterns
~~~~~~~~~~~~~~~~
* ``except SomeError: pass`` inside ``TYPE_CHECKING`` or ``if __name__``
  guards (not reachable at runtime).
* ``except SomeError:`` followed by at least one statement that is *not*
  a bare ``pass``.
* Files outside ``src/vamos/`` (tests, scripts, notebooks, examples) are
  not checked.
"""

from __future__ import annotations

import ast
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_bare_except(handler: ast.ExceptHandler) -> bool:
    """Return True when the handler has no exception type (bare ``except:``)."""
    return handler.type is None


def _is_silent_swallow(handler: ast.ExceptHandler) -> bool:
    """Return True when the handler body is a single ``pass`` statement."""
    if len(handler.body) != 1:
        return False
    stmt = handler.body[0]
    return isinstance(stmt, ast.Pass)


class _ExceptScanner(ast.NodeVisitor):
    """Walk an AST and collect problematic exception handlers."""

    def __init__(self, rel_path: str) -> None:
        self.rel_path = rel_path
        self.violations: list[str] = []

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if _is_bare_except(node):
            self.violations.append(
                f"{self.rel_path}:{node.lineno}: bare except: (always specify an exception type)"
            )
        elif _is_silent_swallow(node):
            # Get the exception type name for the message
            exc_name = ast.dump(node.type) if node.type else "?"
            self.violations.append(
                f"{self.rel_path}:{node.lineno}: silent swallow (except ... : pass) -- "
                f"add logging, re-raise, or a comment explaining why silence is safe"
            )
        self.generic_visit(node)


def test_no_bare_except_in_library() -> None:
    """Scan src/vamos/ for bare except: and silently swallowed exceptions."""
    repo_root = _repo_root()
    src_root = repo_root / "src" / "vamos"
    violations: list[str] = []

    for path in sorted(src_root.rglob("*.py")):
        rel_path = path.relative_to(repo_root).as_posix()
        text = path.read_text(encoding="utf-8-sig")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue  # other gates handle parse errors
        scanner = _ExceptScanner(rel_path)
        scanner.visit(tree)
        violations.extend(scanner.violations)

    if violations:
        header = [
            "Exception handling violations detected:",
            "",
            "Rules:",
            "  1. Never use bare 'except:' -- always specify an exception type.",
            "  2. Never silently swallow exceptions with 'except ...: pass'.",
            "     Use logging.debug(..., exc_info=True) or add a # reason comment.",
            "",
        ]
        header.extend(f"- {v}" for v in sorted(violations))
        raise AssertionError("\n".join(header))
