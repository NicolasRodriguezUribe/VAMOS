from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def read_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def groupby(rows: List[Dict[str, Any]], keys: Tuple[str, ...]) -> Dict[Tuple[str, ...], List[Dict[str, Any]]]:
    out: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        k = tuple(str(row.get(key, "")) for key in keys)
        out.setdefault(k, []).append(row)
    return out


def median_iqr(vals: List[float]) -> Tuple[float, float]:
    v = np.array([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan")
    med = float(np.median(v))
    q1 = float(np.quantile(v, 0.25))
    q3 = float(np.quantile(v, 0.75))
    return med, (q3 - q1)


def write_latex_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    keys = ("problem", "algorithm", "engine", "variant")
    grouped = groupby(rows, keys)

    problems = sorted({r["problem"] for r in rows})
    algos = sorted({r["algorithm"] for r in rows})
    engines = sorted({r["engine"] for r in rows})
    variants = sorted({r["variant"] for r in rows})

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{HV/IGD$^+$ and runtime summary for stopping/archive variants (median $\pm$ IQR).}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\begin{tabular}{l l l l c c c}")
    lines.append(r"\toprule")
    lines.append(r"Problem & Algo & Engine & Variant & HV & IGD$^+$ & Time (s) \\")
    lines.append(r"\midrule")

    for problem in problems:
        for algo in algos:
            for engine in engines:
                for variant in variants:
                    key = (problem, algo, engine, variant)
                    if key not in grouped:
                        continue
                    rr = grouped[key]
                    hv = [to_float(x.get("hv_final")) for x in rr]
                    igd = [to_float(x.get("igd_plus")) for x in rr]
                    t = [to_float(x.get("runtime_s")) for x in rr]
                    hv_m, hv_i = median_iqr(hv)
                    ig_m, ig_i = median_iqr(igd)
                    t_m, t_i = median_iqr(t)
                    lines.append(
                        f"{problem} & {algo} & {engine} & {variant} & "
                        f"${hv_m:.4g}_{{{hv_i:.2g}}}$ & "
                        f"${ig_m:.4g}_{{{ig_i:.2g}}}$ & "
                        f"${t_m:.4g}_{{{t_i:.2g}}}$ \\\\"
                    )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_tradeoff(path: Path, rows: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for plotting.") from exc

    def key_base(row: Dict[str, Any]) -> Tuple[str, str, str, str]:
        return (row["problem"], row["algorithm"], row["engine"], str(row.get("seed", "")))

    baseline = {key_base(r): r for r in rows if r["variant"] == "baseline"}

    xs = []
    ys = []
    for r in rows:
        if r["variant"] == "baseline":
            continue
        key = key_base(r)
        hv = to_float(r.get("hv_final"))
        rt = to_float(r.get("runtime_s"))
        if key in baseline:
            hv0 = to_float(baseline[key].get("hv_final"))
            rt0 = to_float(baseline[key].get("runtime_s"))
            xs.append(rt - rt0)
            ys.append(hv - hv0)
        else:
            xs.append(rt)
            ys.append(hv)

    plt.figure()
    plt.scatter(xs, ys)
    plt.xlabel("Runtime delta vs baseline (s)")
    plt.ylabel("HV delta vs baseline")
    plt.title("Stopping/Archive trade-off (per run)")
    plt.grid(True, alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def write_section(path: Path, table_tex: str, fig_path: str) -> None:
    txt = rf"""
\section{{HV-based stopping and bounded archive}}

We evaluate method variants that enable (i) a bounded archive with explicit pruning contracts and (ii) HV-based
convergence stopping driven by a sampled HV trace. Each run emits \texttt{{hv\_trace.csv}} and
\texttt{{archive\_stats.csv}} and records stopping/archive decisions in \texttt{{metadata.json}}.

\paragraph{{Summary.}} Table~\ref{{tab:hv-archive-summary}} reports HV, IGD$^+$, and runtime for each variant.
Figure~\ref{{fig:hv-archive-tradeoff}} visualizes the runtime--quality trade-off relative to the baseline.

\input{{{table_tex}}}

\begin{{figure}}[t]
\centering
\includegraphics[width=0.85\linewidth]{{{fig_path}}}
\caption{{Runtime vs quality trade-off for stopping/archive variants.}}
\label{{fig:hv-archive-tradeoff}}
\end{{figure}}
"""
    path.write_text(txt.strip() + "\n", encoding="utf-8")


def patch_main_include(main_tex: Path, section_rel: str) -> None:
    if not main_tex.exists():
        return
    txt = main_tex.read_text(encoding="utf-8")
    if section_rel in txt:
        return
    marker = r"\input{sections/04_results_engine}"
    if marker in txt:
        txt = txt.replace(marker, marker + "\n" + rf"\input{{{section_rel}}}")
    else:
        txt = txt.replace(r"\end{document}", rf"\input{{{section_rel}}}" + "\n\\end{document}")
    main_tex.write_text(txt, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-csv", required=True)
    ap.add_argument("--paper-root", required=True)
    ap.add_argument("--tag", default="hv_archive_campaign_core")
    args = ap.parse_args()

    metrics = Path(args.metrics_csv).resolve()
    paper = Path(args.paper_root).resolve()

    rows = read_rows(metrics)
    if not rows:
        print("No rows in metrics; abort.")
        return 2

    table_path = paper / "tables" / f"{args.tag}_summary.tex"
    fig_path = paper / "figures" / f"{args.tag}_tradeoff.png"
    sec_path = paper / "sections" / "04b_results_hv_archive.tex"

    write_latex_table(table_path, rows)
    plot_tradeoff(fig_path, rows)

    write_section(
        sec_path,
        table_tex=f"tables/{table_path.name}",
        fig_path=f"figures/{fig_path.name}",
    )

    main_tex = paper / "main.tex"
    patch_main_include(main_tex, "sections/04b_results_hv_archive")

    print("Wrote table:", table_path)
    print("Wrote figure:", fig_path)
    print("Wrote section:", sec_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
