from __future__ import annotations

import shutil
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parents[2]

    tidy = repo / "artifacts" / "tidy" / "engine_smoke.csv"
    if not tidy.exists():
        print("ERROR: tidy input not found:", tidy)
        return 2

    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except Exception as e:
        print("Missing deps. Install: pip install pandas matplotlib")
        raise

    df = pd.read_csv(tidy)

    # Normalize columns defensively
    for col in ["engine", "runtime_seconds", "front_size", "seed", "problem", "algorithm", "suite"]:
        if col not in df.columns:
            df[col] = None

    # ---- Outputs ----
    artifacts_plots = repo / "artifacts" / "plots"
    artifacts_tables = repo / "artifacts" / "tables"
    paper_figures = repo / "paper" / "manuscript" / "figures"
    paper_tables = repo / "paper" / "manuscript" / "tables"
    for d in (artifacts_plots, artifacts_tables, paper_figures, paper_tables):
        d.mkdir(parents=True, exist_ok=True)

    plot_path = artifacts_plots / "engine_smoke_runtime.png"
    table_path = artifacts_tables / "engine_smoke_table.tex"

    # ---- Plot: runtime by engine (smoke) ----
    # Use median runtime per engine (more stable even if later there are multiple seeds)
    # Filter non-null runtimes
    dfr = df.dropna(subset=["runtime_seconds"]).copy()
    if not dfr.empty:
        agg = dfr.groupby("engine", as_index=False)["runtime_seconds"].median().sort_values("runtime_seconds", ascending=True)
        plt.figure()
        plt.bar(agg["engine"].astype(str), agg["runtime_seconds"])
        plt.xlabel("Engine")
        plt.ylabel("Runtime (seconds)")
        title = "Engine smoke run runtime (median)"
        # add context if single suite/problem
        if df["problem"].nunique() == 1 and df["algorithm"].nunique() == 1:
            title += f" — {df['algorithm'].iloc[0]} on {df['problem'].iloc[0]}"
        plt.title(title)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print("Wrote plot:", plot_path.relative_to(repo))
    else:
        print("WARNING: No runtime_seconds found; skipping plot.")

    # ---- Table: compact LaTeX summary ----
    cols = []
    for c in [
        "suite",
        "algorithm",
        "problem",
        "engine",
        "seed",
        "max_evaluations",
        "population_size",
        "runtime_seconds",
        "front_size",
        "fun_ncols",
        "x_ncols",
    ]:
        if c in df.columns:
            cols.append(c)

    tab = df[cols].copy()

    # Round runtime for readability
    if "runtime_seconds" in tab.columns:
        tab["runtime_seconds"] = tab["runtime_seconds"].apply(lambda x: "" if str(x) == "nan" else f"{float(x):.4g}")

    # Ensure ints display nicely
    for ic in ["seed", "max_evaluations", "population_size", "front_size", "fun_ncols", "x_ncols"]:
        if ic in tab.columns:

            def _fmt(v):
                try:
                    if str(v) == "nan":
                        return ""
                    return str(int(float(v)))
                except Exception:
                    return str(v)

            tab[ic] = tab[ic].apply(_fmt)

    # Write a booktabs table fragment (to \input{} in the manuscript)
    lines = []
    lines.append(r"\begin{tabular}{l l l l r r r r r r}")
    lines.append(r"\toprule")
    header = ["suite", "algorithm", "problem", "engine", "seed", "evals", "pop", "runtime(s)", "front", "m", "n"]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    for _, row in tab.iterrows():
        suite = row.get("suite", "")
        algo = row.get("algorithm", "")
        prob = row.get("problem", "")
        eng = row.get("engine", "")
        seed = row.get("seed", "")
        evals = row.get("max_evaluations", "")
        pop = row.get("population_size", "")
        rt = row.get("runtime_seconds", "")
        front = row.get("front_size", "")
        m = row.get("fun_ncols", "")
        n = row.get("x_ncols", "")
        lines.append(f"{suite} & {algo} & {prob} & {eng} & {seed} & {evals} & {pop} & {rt} & {front} & {m} & {n} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote table:", table_path.relative_to(repo))

    # ---- Sync to paper ----
    shutil.copy2(plot_path, paper_figures / plot_path.name)
    shutil.copy2(table_path, paper_tables / table_path.name)
    print("Synced to paper/manuscript/{figures,tables}/")

    # ---- Write results section to include artifacts ----
    results_tex = repo / "paper" / "manuscript" / "sections" / "04_results_engine.tex"
    results_tex.write_text(
        r"""
\section{Engine differentiation: smoke sanity check}

Table~\ref{tab:engine-smoke} reports a minimal end-to-end sanity check where the same NSGA-II configuration
is executed under three engines. This is not intended as a performance claim; it validates that the run
artifact contract is stable across engines and that basic timing/outputs are collected consistently.

\begin{table}[h]
\centering
\caption{Smoke run summary across engines (single seed).}
\label{tab:engine-smoke}
\input{../tables/engine_smoke_table.tex}
\end{table}

\begin{figure}[h]
\centering
\includegraphics[width=0.85\linewidth]{../figures/engine_smoke_runtime.png}
\caption{Smoke run runtime (median per engine).}
\label{fig:engine-smoke-runtime}
\end{figure}
""".lstrip(),
        encoding="utf-8",
    )
    print("Updated:", results_tex.relative_to(repo))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
