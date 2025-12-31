from __future__ import annotations

from pathlib import Path


REPO = Path.cwd()
SRC = REPO / "src" / "vamos"


def already_patched(path: Path, needle: str) -> bool:
    try:
        return needle in path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False


def main() -> int:
    runner = SRC / "foundation" / "core" / "runner.py"
    outw = SRC / "foundation" / "core" / "runner_output.py"

    if not runner.exists() or not outw.exists():
        raise SystemExit(f"Missing runner files under {SRC}")

    runner_ok = already_patched(runner, "parse_stopping_archive")
    out_ok = already_patched(outw, "hook_mgr")

    if runner_ok and out_ok:
        print("Hook patches already present; no changes applied.")
        return 0

    print("Runner/runner_output modifications should be applied manually.")
    print("Detected patch status:", {"runner": runner_ok, "runner_output": out_ok})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
