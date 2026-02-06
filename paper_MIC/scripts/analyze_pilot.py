"""Quick analysis of the AOS policy pilot experiment."""
import csv
import statistics
from collections import defaultdict

existing = list(csv.DictReader(open("experiments/mic/mic_ablation.csv")))
pilot = list(csv.DictReader(open("experiments/mic/pilot_ablation.csv")))

PROBLEMS = ["rwa1", "rwa3", "rwa7", "re41", "rwa9", "re61"]
VARIANTS = ["baseline", "random", "aos", "aos_eps15", "aos_swucb", "aos_ts"]
LABELS = ["Base", "Random", "AOS-orig", "AOS-e15", "SW-UCB", "Thompson"]
NEW_VARS = ["aos_eps15", "aos_swucb", "aos_ts"]
SHORT = {"aos_eps15": "e15", "aos_swucb": "swucb", "aos_ts": "ts"}

data = defaultdict(lambda: defaultdict(list))
rt = defaultdict(lambda: defaultdict(list))
for r in existing + pilot:
    p, v = r["problem"], r["variant"]
    if p in PROBLEMS:
        data[p][v].append(float(r["hypervolume"]))
        rt[p][v].append(float(r["runtime_seconds"]))

# --- HV table ---
print("\nMEDIAN NORMALIZED HV:")
print(f"{'Problem':<10}", end="")
for l in LABELS:
    print(f"  {l:>10}", end="")
print("  | Deltas vs baseline")
print("-" * 120)

for p in PROBLEMS:
    meds = {}
    for v in VARIANTS:
        vals = data[p].get(v, [])
        meds[v] = statistics.median(vals) if vals else float("nan")

    # Find best variant overall
    best_v = max(VARIANTS, key=lambda v: meds.get(v, 0))

    print(f"{p:<10}", end="")
    for v in VARIANTS:
        m = meds[v]
        marker = " *" if v == best_v else "  "
        print(f"{m:10.4f}{marker}", end="")

    # Deltas
    base = meds["baseline"]
    print("  | ", end="")
    for v in NEW_VARS:
        d = meds[v] - base
        print(f"{SHORT[v]}:{d:+.4f} ", end="")
    print()

# --- Summary wins ---
print("\nWINS vs BASELINE (median > baseline + 0.0005):")
for v in NEW_VARS:
    wins = sum(1 for p in PROBLEMS
               if statistics.median(data[p].get(v, [0])) > statistics.median(data[p].get("baseline", [0])) + 0.0005)
    print(f"  {v}: {wins}/6")

print("\nWINS vs AOS-ORIGINAL (median > aos_original + 0.0005):")
for v in NEW_VARS:
    wins = sum(1 for p in PROBLEMS
               if statistics.median(data[p].get(v, [0])) > statistics.median(data[p].get("aos", [0])) + 0.0005)
    print(f"  {v}: {wins}/6")

# --- Runtime ---
print(f"\nMEDIAN RUNTIME (seconds):")
print(f"{'Problem':<10}", end="")
for l in LABELS:
    print(f"  {l:>10}", end="")
print()
print("-" * 90)
for p in PROBLEMS:
    print(f"{p:<10}", end="")
    for v in VARIANTS:
        vals = rt[p].get(v, [])
        m = statistics.median(vals) if vals else float("nan")
        print(f"  {m:10.2f}", end="")
    print()

# --- Operator usage from trace ---
print("\nOPERATOR USAGE (% of pulls that are sbx+pm):")
trace = list(csv.DictReader(open("experiments/mic/pilot_trace.csv")))
for v in NEW_VARS:
    sub = [r for r in trace if r["variant"] == v]
    total = len(sub)
    sbx = sum(1 for r in sub if r["op_name"] == "sbx+pm")
    pct = 100 * sbx / total if total > 0 else 0
    print(f"  {v}: {pct:.1f}% sbx+pm ({sbx}/{total})")
    # Per-problem
    for p in PROBLEMS:
        psub = [r for r in sub if r["problem"] == p]
        pt = len(psub)
        ps = sum(1 for r in psub if r["op_name"] == "sbx+pm")
        pp = 100 * ps / pt if pt > 0 else 0
        print(f"    {p}: {pp:.1f}%")
