from __future__ import annotations

import shutil
from pathlib import Path

def main() -> int:
    repo = Path.cwd()
    tidy = repo / "artifacts" / "tidy" / "engine_study_full_continuous.csv"
    if not tidy.exists():
        print("ERROR: missing tidy:", tidy)
        return 2

    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(tidy)

    # Output dirs
    artifacts_plots  = repo / "artifacts" / "plots"
    artifacts_tables = repo / "artifacts" / "tables"
    paper_figures    = repo / "paper" / "manuscript" / "figures"
    paper_tables     = repo / "paper" / "manuscript" / "tables"
    for d in (artifacts_plots, artifacts_tables, paper_figures, paper_tables):
        d.mkdir(parents=True, exist_ok=True)

    # Slice report name
    table_path = artifacts_tables / "engine_full_continuous_slice.tex"
    plot_path  = artifacts_plots  / "engine_full_continuous_slice_runtime.png"

    # Table: one row per run (compact)
    cols = ["suite","algorithm","problem","engine","seed","max_evaluations","population_size","runtime_seconds","front_size","fun_ncols"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    tab = df[cols].copy()

    # Formatting
    tab["runtime_seconds"] = tab["runtime_seconds"].apply(lambda x: "" if str(x) == "nan" else f"{float(x):.4g}")
    for ic in ["seed","max_evaluations","population_size","front_size","fun_ncols"]:
        tab[ic] = tab[ic].apply(lambda v: "" if str(v) == "nan" else str(int(float(v))))

    lines = []
    lines.append(r"\begin{tabular}{l l l l r r r r r r}")
    lines.append(r"\toprule")
    lines.append(r"suite & algo & problem & engine & seed & evals & pop & rt(s) & front & m \\")
    lines.append(r"\midrule")
    for _, r in tab.iterrows():
        lines.append(
            f"{r['suite']} & {r['algorithm']} & {r['problem']} & {r['engine']} & "
            f"{r['seed']} & {r['max_evaluations']} & {r['population_size']} & {r['runtime_seconds']} & {r['front_size']} & {r['fun_ncols']} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote table:", table_path.relative_to(repo))

    # Plot: runtime over seed
    dfr = df.dropna(subset=["runtime_seconds"]).copy()
    if not dfr.empty:
        dfr = dfr.sort_values("seed")
        plt.figure()
        plt.plot(dfr["seed"], dfr["runtime_seconds"], marker="o")
        plt.xlabel("Seed")
        plt.ylabel("Runtime (seconds)")
        plt.title("Validation slice runtime (6 runs)")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print("Wrote plot:", plot_path.relative_to(repo))

    # Sync
    shutil.copy2(table_path, paper_tables / table_path.name)
    shutil.copy2(plot_path, paper_figures / plot_path.name)
    print("Synced to paper/manuscript/{tables,figures}/")

    # Append a short subsection to results (without duplicating)
    results_tex = repo / "paper" / "manuscript" / "sections" / "04_results_engine.tex"
    old = results_tex.read_text(encoding="utf-8") if results_tex.exists() else ""
    tag = "Validation slice (continuous full campaign)"
    if tag not in old:
        results_tex.write_text(
            old.rstrip()
            + "\n\n"
            + r"""\subsection{Validation slice (continuous full campaign)}

Before launching the full campaign, we executed a small slice (6 runs) to validate that the generated
configurations run end-to-end, produce stable artifacts, and can be collected into tidy tables.

\begin{table}[h]
\centering
\caption{Validation slice runs (6).}
\label{tab:engine-full-slice}
\input{../tables/engine_full_continuous_slice.tex}
\end{table}

\begin{figure}[h]
\centering
\includegraphics[width=0.75\linewidth]{../figures/engine_full_continuous_slice_runtime.png}
\caption{Validation slice runtime over seeds.}
\label{fig:engine-full-slice-rt}
\end{figure}
""",
            encoding="utf-8",
        )
        print("Updated:", results_tex.relative_to(repo))
    else:
        print("Results section already contains validation slice; not duplicating.")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
