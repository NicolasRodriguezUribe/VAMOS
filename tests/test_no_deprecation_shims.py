from __future__ import annotations

from pathlib import Path


FORBIDDEN_NAMES = {"compat", "shims", "legacy"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_no_deprecation_shims() -> None:
    repo_root = _repo_root()
    src_root = repo_root / "src" / "vamos"
    deprecation_hits: list[str] = []
    forbidden_paths: list[str] = []

    for path in src_root.rglob("*.py"):
        rel_path = path.relative_to(repo_root).as_posix()
        parts = Path(rel_path).parts
        for part in parts:
            if part in FORBIDDEN_NAMES or Path(part).stem in FORBIDDEN_NAMES:
                forbidden_paths.append(rel_path)
                break

        text = path.read_text(encoding="utf-8-sig")
        for idx, line in enumerate(text.splitlines(), start=1):
            if "DeprecationWarning" in line:
                deprecation_hits.append(f"{rel_path}:{idx}: {line.strip()}")

    errors: list[str] = []
    if forbidden_paths:
        errors.append("Forbidden compat/shim/legacy paths found:")
        errors.extend(f"- {path}" for path in sorted(set(forbidden_paths)))
    if deprecation_hits:
        errors.append("DeprecationWarning usage found:")
        errors.extend(f"- {hit}" for hit in deprecation_hits)

    if errors:
        raise AssertionError("\n".join(errors))
