from __future__ import annotations

from pathlib import Path

REPO = Path.cwd()
SRC = REPO / "src" / "vamos"

TERMS = [
    "archive_type",
    "archive",
    "BoundedArchive",
    "Archive",
    "hv",
    "hypervolume",
    "HV",
    "stopping",
    "early",
    "monitor",
    "trace",
    "metadata.json",
    "resolved_config.json",
]


def scan_terms() -> list[tuple[str, int, str]]:
    hits = []
    for p in SRC.rglob("*.py"):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for i, line in enumerate(txt.splitlines(), start=1):
            for t in TERMS:
                if t in line:
                    hits.append((str(p.relative_to(REPO)), i, line.strip()))
                    break
    return hits


def main() -> int:
    print("Repo:", REPO)
    print("Src:", SRC)
    hits = scan_terms()
    print("\nTotal hits:", len(hits))

    # group by file and print top snippets
    by = {}
    for f, ln, line in hits:
        by.setdefault(f, []).append((ln, line))

    # show top 25 files by hit count
    files = sorted(by.items(), key=lambda kv: -len(kv[1]))
    print("\nTop files by hits:")
    for f, items in files[:25]:
        print(f"  {len(items):4d}  {f}")

    # print first 120 concrete matches (actionable)
    print("\nFirst matches (up to 120):")
    c = 0
    for f, items in files:
        for ln, line in items[:8]:
            print(f"{f}:{ln}: {line}")
            c += 1
            if c >= 120:
                return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
