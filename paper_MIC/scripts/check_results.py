"""Check current AOS results: main experiment + pilot comparison."""
import csv
import statistics
from collections import defaultdict

main = list(csv.DictReader(open("experiments/mic/mic_ablation.csv")))
pilot = list(csv.DictReader(open("experiments/mic/pilot_ablation.csv")))

ALL_PROBLEMS = [
    "re21", "re24", "rwa1",
    "re31", "re32", "re33", "re34", "re37",
    "rwa2", "rwa3", "rwa4", "rwa5", "rwa6", "rwa7",
    "rwa8", "re41", "re42",
    "rwa9", "rwa10", "re61", "re91",
]

NOBJ = {
    "re21": 2, "re24": 2, "rwa1": 2,
    "re31": 3, "re32": 3, "re33": 3, "re34": 3, "re37": 3,
    "rwa2": 3, "rwa3": 3, "rwa4": 3, "rwa5": 3, "rwa6": 3, "rwa7": 3,
    "rwa8": 4, "re41": 4, "re42": 4,
    "rwa9": 5, "rwa10": 7, "re61": 6, "re91": 9,
}

PILOT_PROBLEMS = {"rwa1", "rwa3", "rwa7", "re41", "rwa9", "re61"}

data = defaultdict(lambda: defaultdict(list))
for r in main + pilot:
    data[r["problem"]][r["variant"]].append(float(r["hypervolume"]))

def med(problem, variant):
    vals = data[problem].get(variant, [])
    return statistics.median(vals) if vals else float("nan")

# ---- Part 1: Current main experiment ----
print("=" * 90)
print("PART 1: MAIN EXPERIMENT (baseline vs random vs AOS-original)")
print("=" * 90)
print(f"{'Problem':<10} {'nObj':>4}  {'Base':>8}  {'Random':>8}  {'AOS':>8}  {'D(b)':>8}  {'D(r)':>8}")
print("-" * 70)

w_b, l_b, w_r, l_r = 0, 0, 0, 0
for p in ALL_PROBLEMS:
    b, r, a = med(p, "baseline"), med(p, "random"), med(p, "aos")
    db, dr = a - b, a - r
    if db > 0.0005: w_b += 1
    elif db < -0.0005: l_b += 1
    if dr > 0.0005: w_r += 1
    elif dr < -0.0005: l_r += 1
    flag = ""
    if NOBJ[p] >= 5:
        flag = " <-- many-obj"
    print(f"{p:<10} {NOBJ[p]:>4}  {b:8.4f}  {r:8.4f}  {a:8.4f}  {db:+8.4f}  {dr:+8.4f}{flag}")

t_b = 21 - w_b - l_b
t_r = 21 - w_r - l_r
print()
print(f"AOS vs Baseline:  {w_b}W / {t_b}T / {l_b}L  (median wins/ties/losses)")
print(f"AOS vs Random:    {w_r}W / {t_r}T / {l_r}L")

# ---- Part 2: Pilot comparison on 6 problems ----
print()
print("=" * 90)
print("PART 2: PILOT (6 problems, comparing 4 AOS policies)")
print("=" * 90)
variants = ["baseline", "aos", "aos_eps15", "aos_swucb", "aos_ts"]
labels = ["Base", "AOS-orig", "e-greedy15", "SW-UCB", "Thompson"]
print(f"{'Problem':<10} {'nObj':>4}", end="")
for l in labels:
    print(f"  {l:>10}", end="")
print()
print("-" * 80)

for p in sorted(PILOT_PROBLEMS, key=ALL_PROBLEMS.index):
    print(f"{p:<10} {NOBJ[p]:>4}", end="")
    meds = {v: med(p, v) for v in variants}
    best = max(meds.values())
    for v in variants:
        m = meds[v]
        star = " *" if abs(m - best) < 1e-6 else "  "
        print(f"  {m:8.4f}{star}", end="")
    print()

# Best new variant summary
print()
print("BEST NEW VARIANT per problem:")
new_vars = ["aos_eps15", "aos_swucb", "aos_ts"]
new_labels = {"aos_eps15": "e-greedy(0.15)", "aos_swucb": "SW-UCB", "aos_ts": "Thompson"}
for p in sorted(PILOT_PROBLEMS, key=ALL_PROBLEMS.index):
    b = med(p, "baseline")
    a_orig = med(p, "aos")
    best_v = max(new_vars, key=lambda v: med(p, v))
    best_m = med(p, best_v)
    print(f"  {p:<10} {new_labels[best_v]:<15}  "
          f"HV={best_m:.4f}  vs base: {best_m - b:+.4f}  "
          f"vs AOS-orig: {best_m - a_orig:+.4f}")

# ---- Part 3: Operator usage from traces ----
print()
print("=" * 90)
print("PART 3: OPERATOR USAGE (% sbx+pm)")
print("=" * 90)
# Original AOS trace
orig_trace = list(csv.DictReader(open("experiments/mic/mic_trace.csv")))
orig_aos = [r for r in orig_trace if r["variant"] == "aos"]
if orig_aos:
    total = len(orig_aos)
    sbx = sum(1 for r in orig_aos if r["op_name"] == "sbx+pm")
    print(f"  AOS-original:  {100*sbx/total:.1f}% sbx+pm  (POLICY COLLAPSE)")

pilot_trace = list(csv.DictReader(open("experiments/mic/pilot_trace.csv")))
for v, label in [("aos_eps15", "e-greedy(0.15)"), ("aos_swucb", "SW-UCB"), ("aos_ts", "Thompson")]:
    sub = [r for r in pilot_trace if r["variant"] == v]
    total = len(sub)
    sbx = sum(1 for r in sub if r["op_name"] == "sbx+pm")
    pct = 100 * sbx / total if total > 0 else 0
    adapt = "moderate" if 40 < pct < 80 else ("low" if pct >= 80 else "aggressive")
    print(f"  {label:<15} {pct:.1f}% sbx+pm  ({adapt} adaptation)")
