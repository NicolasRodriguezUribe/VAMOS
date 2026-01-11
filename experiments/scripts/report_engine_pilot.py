from __future__ import annotations

import ast
import re
import shutil
from pathlib import Path


def q25(x):
    return x.quantile(0.25)


def q75(x):
    return x.quantile(0.75)


def safe_slug(val: object) -> str:
    s = "" if val is None else str(val)
    s = s.strip()
    if not s:
        return "unknown"
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def normalize_problem(problem_val: object, suite_val: object) -> tuple[str, str]:
    label = "" if problem_val is None else str(problem_val)
    key = None
    if isinstance(problem_val, dict):
        label = str(problem_val.get("label") or problem_val.get("key") or label)
        key = problem_val.get("key")
    else:
        s = str(problem_val)
        if s.strip().startswith("{") and "key" in s:
            try:
                d = ast.literal_eval(s)
                if isinstance(d, dict):
                    label = str(d.get("label") or d.get("key") or label)
                    key = d.get("key")
            except Exception:
                pass
    if not key and suite_val is not None and str(suite_val).strip():
        key = str(suite_val).lower()
    if not key:
        key = safe_slug(label).lower()
    if not label and suite_val is not None:
        label = str(suite_val)
    return label, str(key)


def main() -> int:
    repo = Path.cwd()

    tidy = repo / "artifacts" / "tidy" / "engine_pilot.csv"
    if not tidy.exists():
        print("ERROR: tidy input not found:", tidy)
        return 2

    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except Exception:
        print("Missing deps. Install: pip install pandas matplotlib")
        return 3

    df = pd.read_csv(tidy)

    # Required cols (defensive)
    for c in [
        "engine",
        "problem",
        "algorithm",
        "suite",
        "seed",
        "runtime_seconds",
        "front_size",
        "max_evaluations",
        "population_size",
        "fun_ncols",
    ]:
        if c not in df.columns:
            df[c] = None

    # Output dirs
    artifacts_plots = repo / "artifacts" / "plots"
    artifacts_tables = repo / "artifacts" / "tables"
    paper_figures = repo / "paper" / "manuscript" / "figures"
    paper_tables = repo / "paper" / "manuscript" / "tables"
    for d in (artifacts_plots, artifacts_tables, paper_figures, paper_tables):
        d.mkdir(parents=True, exist_ok=True)

    # Aggregate over seeds: median + IQR per (problem, engine)
    dfr = df.dropna(subset=["runtime_seconds"]).copy()
    grp = dfr.groupby(["suite", "algorithm", "problem", "engine"], as_index=False).agg(
        runtime_med=("runtime_seconds", "median"),
        runtime_q25=("runtime_seconds", q25),
        runtime_q75=("runtime_seconds", q75),
        front_med=("front_size", "median"),
        m=("fun_ncols", "max"),
        evals=("max_evaluations", "max"),
        pop=("population_size", "max"),
        seeds=("seed", lambda s: ",".join(str(int(x)) for x in sorted(set(s.dropna())))),
    )

    # Speedup vs numpy within each (suite,algo,problem)
    base = grp[grp["engine"].astype(str).str.lower().eq("numpy")][["suite", "algorithm", "problem", "runtime_med"]].rename(
        columns={"runtime_med": "runtime_numpy"}
    )
    merged = grp.merge(base, on=["suite", "algorithm", "problem"], how="left")
    merged["speedup_vs_numpy"] = merged["runtime_numpy"] / merged["runtime_med"]

    # ---- Plot 1: runtime bars per problem (median) ----
    # We generate one figure per problem to keep it readable.
    plot_paths = []
    for (suite, algo, prob), sub in merged.groupby(["suite", "algorithm", "problem"]):
        suite_str = "" if suite is None else str(suite)
        algo_str = "" if algo is None else str(algo)
        prob_label, prob_key = normalize_problem(prob, suite_str)
        suite_slug = safe_slug(suite_str)
        algo_slug = safe_slug(algo_str)
        prob_slug = safe_slug(prob_key)

        sub = sub.sort_values("runtime_med", ascending=True)
        plt.figure()
        plt.bar(sub["engine"].astype(str), sub["runtime_med"])
        plt.xlabel("Engine")
        plt.ylabel("Runtime (seconds) - median over seeds")
        if suite_str and suite_str != prob_label:
            title = f"Pilot runtime - {algo_str} on {prob_label} ({suite_str})"
        else:
            title = f"Pilot runtime - {algo_str} on {prob_label}"
        plt.title(title)
        plt.tight_layout()
        p = artifacts_plots / f"engine_pilot_runtime__{suite_slug}__{algo_slug}__{prob_slug}.png"
        plt.savefig(p, dpi=200)
        plt.close()
        plot_paths.append(p)
        print("Wrote plot:", p.relative_to(repo))

    # ---- Plot 2: speedup vs numpy per problem ----
    speed_paths = []
    for (suite, algo, prob), sub in merged.groupby(["suite", "algorithm", "problem"]):
        suite_str = "" if suite is None else str(suite)
        algo_str = "" if algo is None else str(algo)
        prob_label, prob_key = normalize_problem(prob, suite_str)
        suite_slug = safe_slug(suite_str)
        algo_slug = safe_slug(algo_str)
        prob_slug = safe_slug(prob_key)

        sub = sub.sort_values("speedup_vs_numpy", ascending=False)
        plt.figure()
        plt.bar(sub["engine"].astype(str), sub["speedup_vs_numpy"])
        plt.xlabel("Engine")
        plt.ylabel("Speedup vs NumPy (median runtime)")
        if suite_str and suite_str != prob_label:
            title = f"Pilot speedup vs NumPy - {algo_str} on {prob_label} ({suite_str})"
        else:
            title = f"Pilot speedup vs NumPy - {algo_str} on {prob_label}"
        plt.title(title)
        plt.tight_layout()
        p = artifacts_plots / f"engine_pilot_speedup__{suite_slug}__{algo_slug}__{prob_slug}.png"
        plt.savefig(p, dpi=200)
        plt.close()
        speed_paths.append(p)
        print("Wrote plot:", p.relative_to(repo))

    # ---- Table: compact LaTeX fragment ----
    table_path = artifacts_tables / "engine_pilot_summary.tex"
    cols = [
        "suite",
        "algorithm",
        "problem",
        "engine",
        "seeds",
        "evals",
        "pop",
        "m",
        "runtime_med",
        "runtime_q25",
        "runtime_q75",
        "speedup_vs_numpy",
        "front_med",
    ]
    tab = merged[cols].copy()

    def fmt(x, nd=4):
        try:
            if x is None or (isinstance(x, float) and (x != x)):
                return ""
            return f"{float(x):.{nd}g}"
        except Exception:
            return str(x)

    lines = []
    lines.append(r"\begin{tabular}{l l l l l r r r r r r r r}")
    lines.append(r"\toprule")
    lines.append(r"suite & algo & problem & engine & seeds & evals & pop & m & rt$_{50}$ & rt$_{25}$ & rt$_{75}$ & spd & front$_{50}$ \\")
    lines.append(r"\midrule")
    for _, r in tab.iterrows():
        lines.append(
            f"{r['suite']} & {r['algorithm']} & {r['problem']} & {r['engine']} & {r['seeds']} & "
            f"{int(r['evals'])} & {int(r['pop'])} & {int(r['m'])} & "
            f"{fmt(r['runtime_med'])} & {fmt(r['runtime_q25'])} & {fmt(r['runtime_q75'])} & "
            f"{fmt(r['speedup_vs_numpy'])} & {fmt(r['front_med'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote table:", table_path.relative_to(repo))

    # ---- Sync to paper ----
    for p in plot_paths + speed_paths:
        shutil.copy2(p, paper_figures / p.name)
    shutil.copy2(table_path, paper_tables / table_path.name)
    print("Synced pilot plots/tables to paper/manuscript/{figures,tables}/")

    # ---- Append to results section (keep smoke section above) ----
    results_tex = repo / "paper" / "manuscript" / "sections" / "04_results_engine.tex"
    old = results_tex.read_text(encoding="utf-8") if results_tex.exists() else ""
    appendix = r"""
\subsection{Pilot engine study (multi-seed)}

We extend the smoke sanity check to a small multi-seed pilot over two benchmark problems and three engines.
We report median runtime and interquartile range (IQR) across seeds, and speedup relative to NumPy.

\begin{table}[h]
\centering
\caption{Pilot summary: median runtime, IQR, and speedup vs. NumPy.}
\label{tab:engine-pilot}
\input{../tables/engine_pilot_summary.tex}
\end{table}

\noindent
Figures \ref{fig:engine-pilot-runtime}--\ref{fig:engine-pilot-speedup} show per-problem runtime and speedup.

\begin{figure}[h]
\centering
\includegraphics[width=0.90\linewidth]{../figures/engine_pilot_runtime__ZDT1__nsgaii__zdt1.png}
\includegraphics[width=0.90\linewidth]{../figures/engine_pilot_runtime__DTLZ2__nsgaii__dtlz2.png}
\caption{Pilot runtime by engine (median over seeds).}
\label{fig:engine-pilot-runtime}
\end{figure}

\begin{figure}[h]
\centering
\includegraphics[width=0.90\linewidth]{../figures/engine_pilot_speedup__ZDT1__nsgaii__zdt1.png}
\includegraphics[width=0.90\linewidth]{../figures/engine_pilot_speedup__DTLZ2__nsgaii__dtlz2.png}
\caption{Pilot speedup vs NumPy (median runtime).}
\label{fig:engine-pilot-speedup}
\end{figure}
"""
    if "Pilot engine study" not in old:
        results_tex.write_text(old.rstrip() + "\n\n" + appendix.lstrip(), encoding="utf-8")
        print("Updated results section:", results_tex.relative_to(repo))
    else:
        print("Results section already contains pilot subsection; not duplicating.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
