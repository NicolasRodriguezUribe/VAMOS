"""
Statistical Analysis Script for VAMOS Paper
============================================
Performs Wilcoxon signed-rank tests and generates statistical comparison tables.

Usage: python run_statistical_tests.py

Reads: experiments/benchmark_paper.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "experiments"
OUTPUT_DIR = Path(__file__).parent / "manuscript"
INPUT_CSV = DATA_DIR / "benchmark_paper.csv"

ALPHA = 0.05  # Significance level

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
        return float('nan'), float('nan')


def compare_frameworks(df, metric, fw1, fw2, *, higher_is_better: bool):
    """Compare two frameworks across all problems using Wilcoxon test."""
    results = []
    
    for problem in df['problem'].unique():
        data1 = df[(df['problem'] == problem) & (df['framework'] == fw1)][metric].values
        data2 = df[(df['problem'] == problem) & (df['framework'] == fw2)][metric].values
        
        if len(data1) != len(data2) or len(data1) == 0:
            continue
        
        p_value, effect = wilcoxon_test(data1, data2)
        
        m1 = float(np.median(data1))
        m2 = float(np.median(data2))
        if higher_is_better:
            winner = fw1 if m1 > m2 else fw2
        else:
            winner = fw1 if m1 < m2 else fw2

        results.append({
            'problem': problem,
            f'{fw1}_median': m1,
            f'{fw2}_median': m2,
            'p_value': p_value,
            'effect_size': effect,
            'significant': (p_value < ALPHA) if np.isfinite(p_value) else False,
            'winner': winner,
        })
    
    return pd.DataFrame(results)


print("\n" + "=" * 60)
print("RUNTIME COMPARISON: VAMOS (numba) vs pymoo")
print("=" * 60)

runtime_comparison = compare_frameworks(
    df, 'runtime_seconds', 'VAMOS (numba)', 'pymoo', higher_is_better=False
)
print(runtime_comparison.to_string())

# Count wins
vamos_wins = (runtime_comparison['winner'] == 'VAMOS (numba)').sum()
pymoo_wins = (runtime_comparison['winner'] == 'pymoo').sum()
significant_count = runtime_comparison['significant'].sum()

print(f"\nVAMOS wins: {vamos_wins}/{len(runtime_comparison)}")
print(f"pymoo wins: {pymoo_wins}/{len(runtime_comparison)}")
print(f"Significant differences: {significant_count}/{len(runtime_comparison)}")

print("\n" + "=" * 60)
print("HYPERVOLUME COMPARISON: VAMOS (numba) vs pymoo")
print("=" * 60)

if 'hypervolume' in df.columns:
    hv_comparison = compare_frameworks(
        df, 'hypervolume', 'VAMOS (numba)', 'pymoo', higher_is_better=True
    )
    print(hv_comparison.to_string())
    
    vamos_hv_wins = (hv_comparison['winner'] == 'VAMOS (numba)').sum()
    pymoo_hv_wins = (hv_comparison['winner'] == 'pymoo').sum()
    
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

backend_comparison = compare_frameworks(
    df, 'runtime_seconds', 'VAMOS (numba)', 'VAMOS (moocore)', higher_is_better=False
)
print(backend_comparison.to_string())

numba_wins = (backend_comparison['winner'] == 'VAMOS (numba)').sum()
moocore_wins = (backend_comparison['winner'] == 'VAMOS (moocore)').sum()
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
):
    """Generate LaTeX table with statistical test results."""
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
        problem = row['problem']
        v1 = float(row[f'{fw1}_median'])
        v2 = float(row[f'{fw2}_median'])
        p = row['p_value']
        sig = r"$\checkmark$" if row['significant'] else ""
        
        # Bold the winner (minimize runtime, maximize quality metrics such as HV).
        if higher_is_better:
            first_wins = v1 > v2
        else:
            first_wins = v1 < v2
        v1_str, v2_str = (f"\\textbf{{{v1:.2f}}}", f"{v2:.2f}") if first_wins else (f"{v1:.2f}", f"\\textbf{{{v2:.2f}}}")
        
        if not np.isfinite(p):
            p_str = "nan"
        else:
            p_str = f"{p:.4f}" if p < 0.0001 else f"{p:.3f}"
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
    caption="Runtime comparison with Wilcoxon signed-rank test results.",
    label="tab:stats_runtime",
    higher_is_better=False,
)
print(latex_runtime)

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

# Overall speedup
vamos_times = df[df['framework'] == 'VAMOS (numba)']['runtime_seconds']
pymoo_times = df[df['framework'] == 'pymoo']['runtime_seconds']

print(f"\nVAMOS (numba) median runtime: {vamos_times.median():.2f}s")
print(f"pymoo median runtime: {pymoo_times.median():.2f}s")
print(f"Overall speedup: {pymoo_times.median() / vamos_times.median():.1f}x")

# By family
def get_family(p):
    if p.startswith('zdt'): return 'ZDT'
    if p.startswith('dtlz'): return 'DTLZ'
    if p.startswith('wfg'): return 'WFG'
    return 'Other'

df['family'] = df['problem'].apply(get_family)

print("\nSpeedup by family:")
for family in ['ZDT', 'DTLZ', 'WFG']:
    v = df[(df['framework'] == 'VAMOS (numba)') & (df['family'] == family)]['runtime_seconds'].median()
    p = df[(df['framework'] == 'pymoo') & (df['family'] == family)]['runtime_seconds'].median()
    if v > 0:
        print(f"  {family}: {p/v:.1f}x")

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
    result = subprocess.run(
        ['pdflatex', '-interaction=nonstopmode', tex_path.name],
        cwd=tex_path.parent,
        capture_output=True, text=True
    )
    for ext in ['.aux', '.log', '.out']:
        try:
            (tex_path.parent / (tex_path.stem + ext)).unlink()
        except:
            pass
    return result.returncode == 0


# Generate HV table if available
if 'hypervolume' in df.columns:
    latex_hv = generate_latex_stats_table(
        hv_comparison,
        fw1="VAMOS (numba)",
        fw2="pymoo",
        caption="Normalized hypervolume comparison with Wilcoxon signed-rank test results.",
        label="tab:stats_hypervolume",
        higher_is_better=True,
    )
else:
    latex_hv = ""

print("\n" + "=" * 60)
print("UPDATING MAIN.TEX")
print("=" * 60)

if MAIN_TEX.exists():
    content = MAIN_TEX.read_text(encoding='utf-8')
    original_len = len(content)
    
    # Replace statistical tables
    content, replaced_runtime = replace_table_in_tex(content, 'tab:stats_runtime', latex_runtime)
    replaced_hv = False
    if latex_hv:
        content, replaced_hv = replace_table_in_tex(content, 'tab:stats_hypervolume', latex_hv)
    
    if replaced_runtime or replaced_hv:
        if len(content) >= original_len * 0.9:
            MAIN_TEX.write_text(content, encoding='utf-8')
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
