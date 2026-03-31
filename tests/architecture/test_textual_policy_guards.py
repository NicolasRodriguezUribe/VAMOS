from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def test_src_vamos_avoids_google_style_args_sections() -> None:
    src_root = ROOT / "src" / "vamos"
    violations = [path.relative_to(ROOT).as_posix() for path in _python_files(src_root) if "Args:" in path.read_text(encoding="utf-8")]
    assert not violations, "Found Google-style 'Args:' docstrings in src/vamos:\n" + "\n".join(violations)


def test_tests_avoid_global_np_random_rand_usage() -> None:
    tests_root = ROOT / "tests"
    banned = "np.random." + "rand("
    violations = [
        path.relative_to(ROOT).as_posix() for path in _python_files(tests_root) if banned in path.read_text(encoding="utf-8")
    ]
    message = "Found global np.random." + "rand() usage in tests:\n"
    assert not violations, message + "\n".join(violations)
