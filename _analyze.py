#!/usr/bin/env python
"""Compare current experiment results with previous run."""
import csv, collections, statistics

ablation = 'experiments/mic/mic_ablation.csv'
rows = []
with open(ablation, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get('hypervolume') == 'hypervolume':
            continue  # skip duplicate header rows
        rows.append(row)

# Group by (problem, variant) -> list of HV values
data = collections.defaultdict(list)
for r in rows:
    key = (r['problem'], r['variant'])
    data[key].append(float(r['hypervolume']))

problems = sorted(set(r['problem'] for r in rows))
variants = ['baseline', 'random', 'aos']

# Previous results (floor_prob=0.05 original run)
prev = {
    'cec2009_uf1':  (0.781, 0.903, 0.902),
    'cec2009_uf2':  (0.930, 0.947, 0.943),
    'cec2009_uf3':  (0.639, 0.717, 0.726),
    'cec2009_uf4':  (0.790, 0.802, 0.801),
    'cec2009_uf5':  (0.350, 0.487, 0.484),
    'cec2009_uf6':  (0.348, 0.422, 0.424),
    'cec2009_uf7':  (0.757, 0.982, 1.000),
    'cec2009_uf8':  (0.279, 0.167, 0.204),
    'cec2009_uf9':  (0.233, 0.111, 0.135),
    'cec2009_uf10': (0.044, 0.005, 0.006),
    'lsmop1':       (0.048, 0.379, 0.350),
    'lsmop2':       (0.699, 0.749, 0.744),
    'lsmop3':       (0.000, 0.009, 0.000),
    'lsmop4':       (0.501, 0.708, 0.725),
    'lsmop5':       (0.000, 0.000, 0.009),
    'lsmop6':       (0.000, 0.000, 0.000),
    'lsmop7':       (0.000, 0.000, 0.000),
    'lsmop8':       (0.000, 0.495, 0.422),
    'lsmop9':       (0.000, 0.000, 0.000),
}

print("=" * 105)
print(f"{'Problem':20s} {'Base':>7s} {'Rand':>7s} {'AOS':>7s} {'D_base':>8s} {'D_rand':>8s} {'prevAOS':>8s} {'chgAOS':>8s} {'Winner':>8s}")
print("=" * 105)

total_wins = 0
total_losses = 0
total_ties = 0

for p in problems:
    vals = {}
    for v in variants:
        hvs = data.get((p, v), [])
        vals[v] = statistics.mean(hvs) if hvs else 0.0

    d_base = vals['aos'] - vals['baseline']
    d_rand = vals['aos'] - vals['random']

    prev_aos = prev.get(p, (0, 0, 0))[2]
    change = vals['aos'] - prev_aos

    if d_base > 0.005:
        winner = 'AOS+'
        total_wins += 1
    elif d_base < -0.005:
        winner = 'BASE+'
        total_losses += 1
    else:
        winner = 'tie'
        total_ties += 1

    print(f"{p:20s} {vals['baseline']:7.3f} {vals['random']:7.3f} {vals['aos']:7.3f} {d_base:+8.3f} {d_rand:+8.3f} {prev_aos:8.3f} {change:+8.3f} {winner:>8s}")

print("=" * 105)
means = {}
for v in variants:
    all_hvs = [statistics.mean(data[(p, v)]) for p in problems if data[(p, v)]]
    means[v] = statistics.mean(all_hvs)
    print(f"  Mean HV {v:10s}: {means[v]:.4f}")

prev_aos_mean = statistics.mean(v[2] for v in prev.values())
prev_base_mean = statistics.mean(v[0] for v in prev.values())
print(f"  Prev AOS mean   : {prev_aos_mean:.4f}")
print(f"  Change in AOS   : {means['aos'] - prev_aos_mean:+.4f}")

print(f"\nAOS vs Baseline: {total_wins} wins, {total_ties} ties, {total_losses} losses")
d_overall = means['aos'] - means['baseline']
print(f"Overall AOS-Base:  {d_overall:+.4f} ({d_overall/max(means['baseline'],0.001)*100:+.1f}%)")

# Focus on the 3 previous loss problems
print("\n--- FOCUS: PREVIOUS LOSS PROBLEMS (UF8/9/10) ---")
for p in ['cec2009_uf8', 'cec2009_uf9', 'cec2009_uf10']:
    vals = {}
    for v in variants:
        hvs = data.get((p, v), [])
        vals[v] = statistics.mean(hvs) if hvs else 0.0
    prev_b, prev_r, prev_a = prev[p]
    d_base = vals['aos'] - vals['baseline']
    prev_d = prev_a - prev_b
    print(f"  {p}:")
    print(f"    NOW:  AOS={vals['aos']:.3f}  Base={vals['baseline']:.3f}  D_base={d_base:+.3f}")
    print(f"    PREV: AOS={prev_a:.3f}  Base={prev_b:.3f}  D_base={prev_d:+.3f}")
    print(f"    AOS change: {vals['aos'] - prev_a:+.3f}")
