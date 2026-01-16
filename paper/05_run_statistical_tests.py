"""
Statistical Analysis Script for VAMOS Paper
============================================
Performs Wilcoxon signed-rank tests and generates statistical comparison tables.

Usage: python paper/05_run_statistical_tests.py

Reads: experiments/benchmark_paper.csv
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import hashlib
from pathlib import Path
from scipy import stats

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "experiments"
OUTPUT_DIR = Path(__file__).parent / "manuscript"
INPUT_CSV = DATA_DIR / "benchmark_paper.csv"

ALPHA = 0.05  # Significance level
HV_EQ_MARGIN_REL = 0.01  # ±1% relative margin for equivalence on normalized HV
BOOTSTRAP_N = 5000  # paired bootstrap replicates for CI
BOOTSTRAP_CI = 0.90  # 90% CI (equivalence via CI-in-margin criterion)

# =============================================================================
# LOAD DATA
# =============================================================================

print("Loading benchmark data...")
df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} results")
print(f"Problems: {df['problem'].nunique()}")
print(f"Seeds: {df['seed'].nunique()}")
print(f"Frameworks: {df['framework'].unique().tolist()}")

# =============================================================================
# WILCOXON SIGNED-RANK TESTS
# =============================================================================


def wilcoxon_test(group1, group2):
    """Perform Wilcoxon signed-rank test, return p-value and effect size."""
    try:
        stat, p_value = stats.wilcoxon(group1, group2)
        # Effect size (r = Z / sqrt(N))
        n = len(group1)
        z = stats.norm.ppf(1 - p_value / 2)
        effect_size = abs(z) / np.sqrt(n)
        return p_value, effect_size
    except Exception:
        return float("nan"), float("nan")


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjustment (controls FWER); NaNs are preserved."""
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)

    finite_idx = np.where(np.isfinite(p))[0]
    if finite_idx.size == 0:
        return out.tolist()

    m = int(finite_idx.size)
    order = finite_idx[np.argsort(p[finite_idx])]

    prev = 0.0
    for k, idx in enumerate(order):
        factor = float(m - k)
        adj = factor * float(p[idx])
        adj = min(1.0, adj)
        adj = max(prev, adj)  # enforce monotonicity
        prev = adj
        out[idx] = adj

    return out.tolist()


def paired_bootstrap_ci(values: np.ndarray, *, stat_fn, ci: float, n_boot: int, seed: int = 0) -> tuple[float, float]:
    """Paired bootstrap CI for a 1D sample (resample indices with replacement)."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        v = float(values[0])
        return v, v

    rng = np.random.default_rng(seed)
    n = int(values.size)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    stats_boot = stat_fn(values[idx], axis=1)
    alpha = 1.0 - float(ci)
    lo, hi = np.quantile(stats_boot, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def stable_seed(*parts: str) -> int:
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\0")
    return int.from_bytes(h.digest(), "little", signed=False)


def compare_frameworks(df, metric, fw1, fw2, *, higher_is_better: bool):
    """Compare two frameworks across all problems using Wilcoxon test."""
    results = []

    for problem in df["problem"].unique():
        data1 = df[(df["problem"] == problem) & (df["framework"] == fw1)][metric].values
        data2 = df[(df["problem"] == problem) & (df["framework"] == fw2)][metric].values

        if len(data1) != len(data2) or len(data1) == 0:
            continue

        p_value, effect = wilcoxon_test(data1, data2)

        m1 = float(np.median(data1))
        m2 = float(np.median(data2))
        if higher_is_better:
            winner = fw1 if m1 > m2 else fw2
        else:
            winner = fw1 if m1 < m2 else fw2

        results.append(
            {
                "problem": problem,
                f"{fw1}_median": m1,
                f"{fw2}_median": m2,
                "p_value_raw": p_value,
                "effect_size": effect,
                "winner": winner,
            }
        )

    return pd.DataFrame(results)


print("\n" + "=" * 60)
print("RUNTIME COMPARISON: VAMOS (numba) vs pymoo")
print("=" * 60)

runtime_comparison = compare_frameworks(df, "runtime_seconds", "VAMOS (numba)", "pymoo", higher_is_better=False)
runtime_comparison["p_value_adj"] = holm_adjust(runtime_comparison["p_value_raw"].tolist())
runtime_comparison["significant"] = runtime_comparison["p_value_adj"].apply(lambda p: bool(np.isfinite(p) and p < ALPHA))
print(runtime_comparison.to_string())

# Count wins
vamos_wins = (runtime_comparison["winner"] == "VAMOS (numba)").sum()
pymoo_wins = (runtime_comparison["winner"] == "pymoo").sum()
significant_count = runtime_comparison["significant"].sum()

print(f"\nVAMOS wins: {vamos_wins}/{len(runtime_comparison)}")
print(f"pymoo wins: {pymoo_wins}/{len(runtime_comparison)}")
print(f"Significant differences: {significant_count}/{len(runtime_comparison)}")

print("\n" + "=" * 60)
print("HYPERVOLUME COMPARISON: VAMOS (numba) vs pymoo")
print("=" * 60)

if "hypervolume" in df.columns:
    hv_comparison = compare_frameworks(df, "hypervolume", "VAMOS (numba)", "pymoo", higher_is_better=True)
    hv_comparison["p_value_adj"] = holm_adjust(hv_comparison["p_value_raw"].tolist())
    hv_comparison["significant"] = hv_comparison["p_value_adj"].apply(lambda p: bool(np.isfinite(p) and p < ALPHA))
    print(hv_comparison.to_string())

    vamos_hv_wins = (hv_comparison["winner"] == "VAMOS (numba)").sum()
    pymoo_hv_wins = (hv_comparison["winner"] == "pymoo").sum()

    print(f"\nHV: VAMOS better: {vamos_hv_wins}/{len(hv_comparison)}")
    print(f"HV: pymoo better: {pymoo_hv_wins}/{len(hv_comparison)}")
else:
    print("No hypervolume data found")

# =============================================================================
# VAMOS BACKEND COMPARISON
# =============================================================================

print("\n" + "=" * 60)
print("BACKEND COMPARISON: numba vs moocore")
print("=" * 60)

backend_comparison = compare_frameworks(df, "runtime_seconds", "VAMOS (numba)", "VAMOS (moocore)", higher_is_better=False)
print(backend_comparison.to_string())

numba_wins = (backend_comparison["winner"] == "VAMOS (numba)").sum()
moocore_wins = (backend_comparison["winner"] == "VAMOS (moocore)").sum()
print(f"\nnumba wins: {numba_wins}/{len(backend_comparison)}")
print(f"moocore wins: {moocore_wins}/{len(backend_comparison)}")

# =============================================================================
# LATEX TABLE GENERATION
# =============================================================================


def generate_latex_stats_table(
    comparison_df,
    *,
    fw1: str,
    fw2: str,
    caption: str,
    label: str,
    higher_is_better: bool,
    value_decimals: int = 2,
):
    """Generate LaTeX table with statistical test results."""

    def _fmt_value(value: float) -> str:
        return f"{value:.{value_decimals}f}"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{l|rr|r|l}",
        r"\toprule",
        rf"\textbf{{Problem}} & \textbf{{{fw1}}} & \textbf{{{fw2}}} & \textbf{{p-value}} & \textbf{{Sig.}} \\",
        r"\midrule",
    ]

    for _, row in comparison_df.iterrows():
        problem = row["problem"]
        v1 = float(row[f"{fw1}_median"])
        v2 = float(row[f"{fw2}_median"])
        p = row.get("p_value_adj", row.get("p_value_raw", float("nan")))
        sig = r"$\checkmark$" if row["significant"] else ""

        # Bold the winner (minimize runtime, maximize quality metrics such as HV).
        if higher_is_better:
            first_wins = v1 > v2
        else:
            first_wins = v1 < v2
        v1_fmt = _fmt_value(v1)
        v2_fmt = _fmt_value(v2)
        v1_str, v2_str = (f"\\textbf{{{v1_fmt}}}", v2_fmt) if first_wins else (v1_fmt, f"\\textbf{{{v2_fmt}}}")

        if not np.isfinite(p):
            p_str = "nan"
        else:
            p_str = "<0.0001" if p < 0.0001 else f"{p:.3f}"
        lines.append(f"{problem} & {v1_str} & {v2_str} & {p_str} & {sig} \\\\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


print("\n" + "=" * 60)
print("LATEX TABLE - Runtime Statistics")
print("=" * 60)
latex_runtime = generate_latex_stats_table(
    runtime_comparison,
    fw1="VAMOS (numba)",
    fw2="pymoo",
    caption="Runtime comparison with Wilcoxon signed-rank test results (Holm-corrected).",
    label="tab:stats_runtime",
    higher_is_better=False,
    value_decimals=2,
)
print(latex_runtime)

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

# Overall speedup
vamos_times = df[df["framework"] == "VAMOS (numba)"]["runtime_seconds"]
pymoo_times = df[df["framework"] == "pymoo"]["runtime_seconds"]

print(f"\nVAMOS (numba) median runtime: {vamos_times.median():.2f}s")
print(f"pymoo median runtime: {pymoo_times.median():.2f}s")
print(f"Overall speedup: {pymoo_times.median() / vamos_times.median():.1f}x")


# By family
def get_family(p):
    if p.startswith("zdt"):
        return "ZDT"
    if p.startswith("dtlz"):
        return "DTLZ"
    if p.startswith("wfg"):
        return "WFG"
    return "Other"


df["family"] = df["problem"].apply(get_family)

print("\nSpeedup by family:")
for family in ["ZDT", "DTLZ", "WFG"]:
    v = df[(df["framework"] == "VAMOS (numba)") & (df["family"] == family)]["runtime_seconds"].median()
    p = df[(df["framework"] == "pymoo") & (df["family"] == family)]["runtime_seconds"].median()
    if v > 0:
        print(f"  {family}: {p / v:.1f}x")

# =============================================================================
# SAVE RESULTS
# =============================================================================

output_csv = DATA_DIR / "statistical_tests.csv"
runtime_comparison.to_csv(output_csv, index=False)
print(f"\nSaved statistical results to {output_csv}")

# =============================================================================
# UPDATE MAIN.TEX
# =============================================================================

import re
import subprocess

MAIN_TEX = OUTPUT_DIR / "main.tex"


def _find_table_bounds(content: str, label: str) -> tuple[int, int, str, str] | None:
    label_token = f"\\label{{{label}}}"
    label_pos = content.find(label_token)
    if label_pos == -1:
        return None

    begin_table = content.rfind(r"\begin{table}", 0, label_pos)
    begin_table_star = content.rfind(r"\begin{table*}", 0, label_pos)
    if begin_table_star > begin_table:
        begin_pos = begin_table_star
        begin_tag = r"\begin{table*}"
        end_tag = r"\end{table*}"
    else:
        begin_pos = begin_table
        begin_tag = r"\begin{table}"
        end_tag = r"\end{table}"

    if begin_pos == -1:
        return None

    end_pos = content.find(end_tag, label_pos)
    if end_pos == -1:
        return None
    end_pos += len(end_tag)

    return begin_pos, end_pos, begin_tag, end_tag


def _normalize_table_env(new_table: str, begin_tag: str, end_tag: str) -> str:
    if begin_tag.endswith("*") and "\\begin{table*}" not in new_table:
        new_table = new_table.replace(r"\begin{table}", begin_tag, 1)
        new_table = new_table.replace(r"\end{table}", end_tag, 1)
    return new_table


def replace_table_in_tex(content: str, label: str, new_table: str) -> tuple[str, bool]:
    """Replace a table in LaTeX content by its label using robust bounds."""
    if not new_table:
        print(f"Skipping {label}: no table content")
        return content, False

    bounds = _find_table_bounds(content, label)
    if not bounds:
        print(f"Warning: Table {label} not found")
        return content, False

    begin_pos, end_pos, begin_tag, end_tag = bounds
    new_table = _normalize_table_env(new_table, begin_tag, end_tag)
    return content[:begin_pos] + new_table + content[end_pos:], True


def compile_latex(tex_path: Path) -> bool:
    """Compile LaTeX to PDF."""
    result = subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_path.name], cwd=tex_path.parent, capture_output=True, text=True)
    for ext in [".aux", ".log", ".out"]:
        try:
            (tex_path.parent / (tex_path.stem + ext)).unlink()
        except:
            pass
    return result.returncode == 0


# Generate HV table if available
if "hypervolume" in df.columns:
    latex_hv = generate_latex_stats_table(
        hv_comparison,
        fw1="VAMOS (numba)",
        fw2="pymoo",
        caption="Normalized hypervolume comparison with Wilcoxon signed-rank test results (Holm-corrected).",
        label="tab:stats_hypervolume",
        higher_is_better=True,
        value_decimals=3,
    )
else:
    latex_hv = ""


# =============================================================================
# EQUIVALENCE + ROBUSTNESS (HV)
# =============================================================================


def _relative_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Relative difference (a-b)/b with epsilon guard."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    eps = 1e-12
    return (a - b) / np.maximum(np.abs(b), eps)


def build_hv_equivalence_details(df_in: pd.DataFrame) -> pd.DataFrame:
    targets = ["pymoo", "DEAP", "jMetalPy", "Platypus"]
    targets = [t for t in targets if t in set(df_in["framework"].unique())]

    problems = sorted(df_in["problem"].unique().tolist())
    rows: list[dict[str, object]] = []

    for fw in targets:
        for problem in problems:
            a = df_in[(df_in["framework"] == "VAMOS (numba)") & (df_in["problem"] == problem)]["hypervolume"].to_numpy()
            b = df_in[(df_in["framework"] == fw) & (df_in["problem"] == problem)]["hypervolume"].to_numpy()
            if a.size == 0 or b.size == 0 or a.size != b.size:
                continue

            rel = _relative_diff(a, b)
            lo, hi = paired_bootstrap_ci(rel, stat_fn=np.median, ci=BOOTSTRAP_CI, n_boot=BOOTSTRAP_N, seed=stable_seed(fw, problem, "hv_eq"))

            eq = bool(np.isfinite(lo) and np.isfinite(hi) and lo >= -HV_EQ_MARGIN_REL and hi <= HV_EQ_MARGIN_REL)
            noninf = bool(np.isfinite(lo) and lo >= -HV_EQ_MARGIN_REL)
            med = float(np.median(rel) * 100.0)

            rows.append(
                {
                    "framework": fw,
                    "problem": problem,
                    "median_delta_percent": med,
                    "ci_low_percent": float(lo * 100.0),
                    "ci_high_percent": float(hi * 100.0),
                    "equivalent": eq,
                    "non_inferior": noninf,
                }
            )

    return pd.DataFrame(rows)


def build_hv_equivalence_summary(df_details: pd.DataFrame) -> pd.DataFrame:
    if df_details.empty:
        return pd.DataFrame(columns=["framework", "n_problems", "eq_problems", "noninf_problems", "median_delta_percent"])

    summary = (
        df_details.groupby("framework")
        .agg(
            n_problems=("problem", "count"),
            eq_problems=("equivalent", "sum"),
            noninf_problems=("non_inferior", "sum"),
            median_delta_percent=("median_delta_percent", "median"),
        )
        .reset_index()
    )

    order = {"pymoo": 0, "DEAP": 1, "jMetalPy": 2, "Platypus": 3}
    summary["__ord"] = summary["framework"].map(lambda x: order.get(str(x), 99))
    summary = summary.sort_values("__ord").drop(columns=["__ord"])
    return summary


def build_hv_robustness_summary(df_in: pd.DataFrame) -> pd.DataFrame:
    frameworks = ["VAMOS (numba)", "pymoo", "DEAP", "jMetalPy", "Platypus"]
    frameworks = [f for f in frameworks if f in set(df_in["framework"].unique())]
    problems = sorted(df_in["problem"].unique().tolist())

    per_fw_iqr: dict[str, list[float]] = {f: [] for f in frameworks}
    per_fw_pct: dict[str, list[float]] = {f: [] for f in frameworks}

    for problem in problems:
        per_fw_vals = {}
        for fw in frameworks:
            vals = df_in[(df_in["framework"] == fw) & (df_in["problem"] == problem)]["hypervolume"].to_numpy(dtype=float)
            if vals.size == 0:
                continue
            per_fw_vals[fw] = vals

        if not per_fw_vals:
            continue

        medians = {fw: float(np.median(vals)) for fw, vals in per_fw_vals.items()}
        best_median = float(max(medians.values()))
        threshold = 0.99 * best_median

        for fw, vals in per_fw_vals.items():
            q1, q3 = np.quantile(vals, [0.25, 0.75])
            iqr = float(q3 - q1)
            pct = float(np.mean(vals >= threshold) * 100.0)
            per_fw_iqr[fw].append(iqr)
            per_fw_pct[fw].append(pct)

    rows = []
    for fw in frameworks:
        iqr_med = float(np.median(per_fw_iqr[fw])) if per_fw_iqr[fw] else float("nan")
        pct_med = float(np.median(per_fw_pct[fw])) if per_fw_pct[fw] else float("nan")
        rows.append({"framework": fw, "median_iqr": iqr_med, "median_pct_near_best": pct_med})

    return pd.DataFrame(rows)


def generate_latex_hv_equivalence_table(df_eq: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Normalized hypervolume equivalence summary (paired bootstrap 90\% CI; equivalence margin $\pm 1\%$ relative). Generated by \texttt{paper/05\_run\_statistical\_tests.py}.}",
        r"\label{tab:hv_equivalence_summary}",
        r"\begin{tabular}{l|rrr}",
        r"\toprule",
        r"\textbf{Framework} & \textbf{Eq.\ problems} & \textbf{Non-inf.\ problems} & \textbf{Median $\Delta$HV (\%)} \\",
        r"\midrule",
    ]

    for _, row in df_eq.iterrows():
        fw = str(row["framework"])
        denom = int(row.get("n_problems", 0) or 0)
        eq = int(row.get("eq_problems", 0) or 0)
        ni = int(row.get("noninf_problems", 0) or 0)
        delta = row.get("median_delta_percent", float("nan"))
        eq_str = f"{eq}/{denom}" if denom else "--"
        ni_str = f"{ni}/{denom}" if denom else "--"
        delta_str = "--" if not np.isfinite(delta) else f"{float(delta):+.2f}"
        lines.append(f"{fw} & {eq_str} & {ni_str} & {delta_str} \\\\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def generate_latex_hv_robustness_table(df_rob: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Robustness summary for normalized hypervolume across seeds (median across problems). ``Near-best'' is defined per problem as HV $\ge 0.99 \cdot \max$ median(HV) among frameworks. Generated by \texttt{paper/05\_run\_statistical\_tests.py}.}",
        r"\label{tab:hv_robustness_summary}",
        r"\begin{tabular}{l|rr}",
        r"\toprule",
        r"\textbf{Framework} & \textbf{Median IQR(HV)} & \textbf{Median \% seeds near-best} \\",
        r"\midrule",
    ]

    for _, row in df_rob.iterrows():
        fw = str(row["framework"])
        iqr = row.get("median_iqr", float("nan"))
        pct = row.get("median_pct_near_best", float("nan"))
        iqr_str = "--" if not np.isfinite(iqr) else f"{float(iqr):.3f}"
        pct_str = "--" if not np.isfinite(pct) else f"{float(pct):.1f}"
        lines.append(f"{fw} & {iqr_str} & {pct_str} \\\\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def generate_latex_hv_equivalence_ci_table(df_details: pd.DataFrame) -> str:
    targets = ["pymoo", "DEAP", "jMetalPy", "Platypus"]
    targets = [t for t in targets if t in set(df_details["framework"].unique())]

    problems = sorted(df_details["problem"].unique().tolist())
    pivot = df_details.pivot_table(index="problem", columns="framework", values=["ci_low_percent", "ci_high_percent"], aggfunc="first")

    lines = [
        r"\begin{table*}[htbp]",
        r"\tiny",
        r"\centering",
        r"\caption{Per-problem paired bootstrap 90\% confidence intervals (CI) for the relative normalized hypervolume difference $\Delta = (\HV_{\VAMOS} - \HV_{\text{fw}})/\HV_{\text{fw}}$, reported in percent. Equivalence margin is $\pm 1\%$. Generated by \texttt{paper/05\_run\_statistical\_tests.py}.}",
        r"\label{tab:hv_equivalence_ci}",
        r"\begin{tabular}{l|" + "r" * len(targets) + "}",
        r"\toprule",
        r"\textbf{Problem} & " + " & ".join([rf"\textbf{{{fw} CI}}" for fw in targets]) + r" \\",
        r"\midrule",
    ]

    def _fmt_ci(lo: float, hi: float) -> str:
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return "--"
        return rf"$[{lo:+.2f},{hi:+.2f}]$"

    for problem in problems:
        cells: list[str] = []
        for fw in targets:
            try:
                lo = float(pivot[("ci_low_percent", fw)].get(problem))
                hi = float(pivot[("ci_high_percent", fw)].get(problem))
            except Exception:
                lo, hi = float("nan"), float("nan")
            cells.append(_fmt_ci(lo, hi))
        lines.append(f"{problem} & " + " & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)

print("\n" + "=" * 60)
print("UPDATING MAIN.TEX")
print("=" * 60)

if MAIN_TEX.exists():
    content = MAIN_TEX.read_text(encoding="utf-8")
    original_len = len(content)

    # Replace statistical tables
    content, replaced_runtime = replace_table_in_tex(content, "tab:stats_runtime", latex_runtime)
    replaced_hv = False
    if latex_hv:
        content, replaced_hv = replace_table_in_tex(content, "tab:stats_hypervolume", latex_hv)

    replaced_eq = False
    replaced_rob = False
    replaced_ci = False
    if "hypervolume" in df.columns:
        eq_details = build_hv_equivalence_details(df)
        eq_summary = build_hv_equivalence_summary(eq_details)
        rob_summary = build_hv_robustness_summary(df)
        latex_eq = generate_latex_hv_equivalence_table(eq_summary)
        latex_rob = generate_latex_hv_robustness_table(rob_summary)
        latex_ci = generate_latex_hv_equivalence_ci_table(eq_details) if not eq_details.empty else ""
        content, replaced_eq = replace_table_in_tex(content, "tab:hv_equivalence_summary", latex_eq)
        content, replaced_rob = replace_table_in_tex(content, "tab:hv_robustness_summary", latex_rob)
        if latex_ci:
            content, replaced_ci = replace_table_in_tex(content, "tab:hv_equivalence_ci", latex_ci)

    if replaced_runtime or replaced_hv or replaced_eq or replaced_rob or replaced_ci:
        if len(content) >= original_len * 0.9:
            MAIN_TEX.write_text(content, encoding="utf-8")
            print(f"main.tex updated ({len(content)} bytes)")

            if compile_latex(MAIN_TEX):
                print(f"PDF compiled: {MAIN_TEX.parent / 'main.pdf'}")
            else:
                print("PDF compilation failed (pdflatex may not be installed)")
        else:
            print("ERROR: Content too short, skipping write")
    else:
        print("No tables updated (labels not found?)")
else:
    print(f"Warning: {MAIN_TEX} not found")
    print("\nAdd these tables to your LaTeX document:")
    print("\n--- Runtime Statistics Table ---")
    print(latex_runtime)
    if latex_hv:
        print("\n--- Hypervolume Statistics Table ---")
        print(latex_hv)

print("\nDone!")
