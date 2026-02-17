"""
MIC Showcase Experiment: AOS Strength Demonstration
====================================================
Runs baseline / random arm / AOS on 6 problems where AOS provides the
most dramatic improvement, then generates a 2×3 convergence-curve figure.

The 6 problems were chosen for maximum visual impact:
  lsmop1    – large-scale (100 vars), AOS +406 % vs baseline
  lsmop8    – baseline HV ≈ 0, AOS achieves 0.252
  dc1dtlz3  – discontinuously constrained, AOS +376 %
  cec2009_uf7 – curved Pareto set, AOS +17 %
  mw5       – constrained, AOS +73 %
  cec2009_uf1 – curved Pareto set, AOS converges to near-optimal rapidly

Usage
-----
  python paper/mic/scripts/04_run_showcase_experiment.py

  # Skip the run, just regenerate the figure from an existing CSV:
  python paper/mic/scripts/04_run_showcase_experiment.py --figure-only

Environment variables
---------------------
  VAMOS_MIC_N_EVALS     Evaluations per run          (default: 100000)
  VAMOS_MIC_N_SEEDS     Seeds per (variant, problem) (default: 30)
  VAMOS_MIC_N_JOBS      Parallel workers             (default: CPU-1)
  VAMOS_MIC_ENGINE      Engine backend               (default: numba)
  VAMOS_MIC_RESUME      1/0 resume into existing CSV (default: 0)
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# ── Path setup ─────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "paper"))
sys.path.insert(0, str(SCRIPTS_DIR))  # allow child processes to import by filename

# Must be set *before* 02_run_mic_experiment.py is loaded (module-level suite select)
os.environ.setdefault("VAMOS_MIC_SUITE", "comprehensive")

# ── Load shared infrastructure from 02_run_mic_experiment.py ───────────────
_MIC_MOD_NAME = "02_run_mic_experiment"
_spec = importlib.util.spec_from_file_location(
    _MIC_MOD_NAME, SCRIPTS_DIR / "02_run_mic_experiment.py"
)
_mic = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules[_MIC_MOD_NAME] = _mic  # register so @dataclass + joblib workers can resolve it
_spec.loader.exec_module(_mic)  # type: ignore[union-attr]

run_single       = _mic.run_single
AOSRuntimeOptions = _mic.AOSRuntimeOptions
_as_int_env      = _mic._as_int_env
_as_str_env      = _mic._as_str_env
_as_bool_env     = _mic._as_bool_env

try:
    from progress_utils import ProgressBar, joblib_progress
except ImportError:
    class ProgressBar:  # type: ignore[no-redef]
        def __init__(self, **_: Any) -> None: pass
        def update(self, _: int = 1) -> None: pass
        def close(self) -> None: pass

    from contextlib import contextmanager

    @contextmanager  # type: ignore[no-redef]
    def joblib_progress(**_: Any):
        yield

# ── Showcase problem list ───────────────────────────────────────────────────
SHOWCASE_PROBLEMS = [
    "lsmop1",         # Row 0: large-scale (100 vars), AOS +406 %
    "lsmop8",         # Row 0: baseline HV ≈ 0, AOS = 0.252
    "dc1dtlz3",       # Row 0: discontinuously constrained, AOS +376 %
    "cec2009_uf7",    # Row 1: curved Pareto set (bi-obj), AOS +17 %
    "mw5",            # Row 1: constrained (bi-obj), AOS +73 %
    "cec2009_uf1",    # Row 1: curved Pareto set, AOS converges fast
]

# Human-readable subplot titles
PROBLEM_LABELS: dict[str, str] = {
    "lsmop1":       "LSMOP1  (+406 %)",
    "lsmop8":       "LSMOP8  (baseline HV ≈ 0)",
    "dc1dtlz3":     "DC1-DTLZ3  (+376 %)",
    "cec2009_uf7":  "UF7  (+17 %)",
    "mw5":          "MW5  (+73 %)",
    "cec2009_uf1":  "UF1",
}

# Variant display: (hex-colour, linestyle, linewidth, legend-label)
VARIANT_STYLES: dict[str, tuple[str, str, float, str]] = {
    "baseline": ("#555555", "-",   1.5, "Baseline (SBX + PM)"),
    "random":   ("#E69F00", "--",  1.5, "Random arm"),
    "aos":      ("#009E73", "-",   2.5, "AOS (original)"),
    "aos_v2":   ("#CC79A7", "-.",  2.0, "AOS v2 (phase-aware, no elim.)"),
}

VARIANTS = ["baseline", "random", "aos", "aos_v2"]

# ── Output paths ────────────────────────────────────────────────────────────
EXP_DIR  = ROOT_DIR / "experiments" / "mic"
FIGS_DIR = ROOT_DIR / "paper" / "mic" / "figures"
CSV_PATH = EXP_DIR / "mic_showcase.csv"
ANY_PATH = EXP_DIR / "mic_showcase_anytime.csv"
FIG_PATH = FIGS_DIR / "showcase_convergence.pdf"


# ── Default AOS options (matching the paper configuration) ─────────────────
def _default_aos_options() -> AOSRuntimeOptions:
    return AOSRuntimeOptions(
        method="thompson_sampling",
        min_usage=5,
        window_size=50,
        floor_prob=0.02,
        elimination_after=300,
        elimination_z=2.0,
        elimination_min_arms=2,
    )


# ── Run the experiment ──────────────────────────────────────────────────────

def run_experiment(
    *,
    n_evals: int,
    n_seeds: int,
    engine: str,
    n_jobs: int,
    resume: bool,
) -> None:
    checkpoints = [5_000, 10_000, 20_000, 50_000]
    if n_evals not in checkpoints:
        checkpoints.append(n_evals)
    checkpoints = sorted(set(c for c in checkpoints if c <= n_evals))

    aos_options = _default_aos_options()

    tasks = [
        (v, p, s)
        for v in VARIANTS
        for p in SHOWCASE_PROBLEMS
        for s in range(n_seeds)
    ]

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    # Resume: skip already-completed (variant, problem, seed) triples
    done: set[tuple[str, str, int]] = set()
    written_main = 0
    written_any = 0

    if not resume:
        for f in [CSV_PATH, ANY_PATH]:
            if f.exists():
                f.unlink()
    else:
        if CSV_PATH.exists():
            existing = pd.read_csv(CSV_PATH)
            written_main = len(existing)
            if {"variant", "problem", "seed"}.issubset(existing.columns):
                existing = existing[existing.get("n_evals", n_evals).astype(int) == n_evals]
                existing["variant"] = existing["variant"].astype(str).str.strip().str.lower()
                existing["problem"] = existing["problem"].astype(str).str.strip().str.lower()
                existing["seed"]    = existing["seed"].astype(int)
                done = set(zip(existing["variant"], existing["problem"], existing["seed"]))
                if done:
                    print(f"Resume: skipping {len(done)} completed runs")
        if ANY_PATH.exists():
            try:
                written_any = len(pd.read_csv(ANY_PATH))
            except Exception:
                written_any = 0

    tasks = [t for t in tasks if t not in done]
    print(f"Total runs: {len(tasks)}  ({len(SHOWCASE_PROBLEMS)} problems × {len(VARIANTS)} variants × {n_seeds} seeds)")
    if not tasks:
        print("Nothing to do — all runs already complete.")
        return

    def _write(rows_main: list[dict], rows_any: list[dict]) -> None:
        nonlocal written_main, written_any
        if rows_main:
            pd.DataFrame(rows_main).to_csv(
                CSV_PATH, mode="a", header=(written_main == 0), index=False
            )
            written_main += len(rows_main)
        if rows_any:
            pd.DataFrame(rows_any).to_csv(
                ANY_PATH, mode="a", header=(written_any == 0), index=False
            )
            written_any += len(rows_any)

    if n_jobs == 1:
        bar = ProgressBar(total=len(tasks), desc="Showcase")
        batch_m: list[dict] = []
        batch_a: list[dict] = []
        for i, (v, p, s) in enumerate(tasks):
            row, chk, _ = run_single(
                v, p, s,
                n_evals=n_evals, engine=engine,
                checkpoints=checkpoints, capture_aos_trace=False,
                aos_options=aos_options,
            )
            batch_m.append(row)
            batch_a.extend(chk)
            # Flush at seed boundaries or end
            is_last = i == len(tasks) - 1
            next_seed = tasks[i + 1][2] if not is_last else None
            if is_last or next_seed != s:
                _write(batch_m, batch_a)
                batch_m, batch_a = [], []
            bar.update(1)
        bar.close()
    else:
        with joblib_progress(total=len(tasks), desc="Showcase"):
            parallel = Parallel(n_jobs=n_jobs, batch_size=1, return_as="generator")
            rows_m: list[dict] = []
            rows_a: list[dict] = []
            for row, chk, _ in parallel(
                delayed(run_single)(
                    v, p, s,
                    n_evals=n_evals, engine=engine,
                    checkpoints=checkpoints, capture_aos_trace=False,
                    aos_options=aos_options,
                )
                for v, p, s in tasks
            ):
                rows_m.append(row)
                rows_a.extend(chk)
            _write(rows_m, rows_a)

    print(f"Results -> {CSV_PATH}  ({written_main} rows)")
    print(f"Anytime -> {ANY_PATH}  ({written_any} rows)")


# ── Generate the figure ────────────────────────────────────────────────────

def make_figure(any_csv: Path, out_pdf: Path) -> None:
    """Convergence curves (mean ± 1 std) for the 6 showcase problems."""
    if not any_csv.exists():
        print(f"Anytime CSV not found: {any_csv}")
        return

    df = pd.read_csv(any_csv)
    df["problem"] = df["problem"].astype(str).str.strip().str.lower()
    df["variant"] = df["variant"].astype(str).str.strip().str.lower()
    df["evals"]   = pd.to_numeric(df["evals"], errors="coerce")
    df["hypervolume"] = pd.to_numeric(df["hypervolume"], errors="coerce")
    df = df.dropna(subset=["evals", "hypervolume"])

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=False)
    fig.suptitle("AOS Showcase: Convergence on Challenging Benchmarks", fontsize=13, y=1.01)

    for ax, problem in zip(axes.flat, SHOWCASE_PROBLEMS):
        sub = df[df["problem"] == problem]
        eval_vals = sorted(sub["evals"].unique())

        for variant, (color, ls, lw, label) in VARIANT_STYLES.items():
            vsub = sub[sub["variant"] == variant]
            means, stds = [], []
            for ev in eval_vals:
                hv_at = vsub[vsub["evals"] == ev]["hypervolume"]
                means.append(hv_at.mean() if len(hv_at) > 0 else np.nan)
                stds.append(hv_at.std(ddof=1) if len(hv_at) > 1 else 0.0)
            means_arr = np.array(means)
            stds_arr  = np.array(stds)
            evals_arr = np.array(eval_vals)

            mask = ~np.isnan(means_arr)
            if mask.sum() == 0:
                continue
            ax.plot(evals_arr[mask], means_arr[mask],
                    color=color, linestyle=ls, linewidth=lw, label=label)
            ax.fill_between(
                evals_arr[mask],
                (means_arr - stds_arr)[mask],
                (means_arr + stds_arr)[mask],
                color=color, alpha=0.12,
            )

        ax.set_title(PROBLEM_LABELS.get(problem, problem), fontsize=10)
        ax.set_xlabel("Function evaluations", fontsize=8)
        ax.set_ylabel("Normalized HV", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K")
        )
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    # Shared legend below the figure
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=4,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=9, frameon=False,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Figure -> {out_pdf}")


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Showcase experiment: AOS vs baseline on 6 high-impact problems"
    )
    ap.add_argument("--figure-only", action="store_true",
                    help="Skip the run; regenerate the figure from an existing CSV.")
    args = ap.parse_args()

    n_evals = _as_int_env("VAMOS_MIC_N_EVALS", 100_000)
    n_seeds = _as_int_env("VAMOS_MIC_N_SEEDS", 30)
    engine  = _as_str_env("VAMOS_MIC_ENGINE", "numba")
    n_jobs  = int(os.environ.get("VAMOS_MIC_N_JOBS", max(1, (os.cpu_count() or 2) - 1)))
    resume  = _as_bool_env("VAMOS_MIC_RESUME", False)

    print("=== AOS Showcase Experiment ===")
    print(f"Problems: {SHOWCASE_PROBLEMS}")
    print(f"Variants: {VARIANTS}")
    print(f"Budget:   {n_evals:,} evals × {n_seeds} seeds")
    print(f"Engine:   {engine}   Workers: {n_jobs}")
    print()

    if not args.figure_only:
        run_experiment(
            n_evals=n_evals,
            n_seeds=n_seeds,
            engine=engine,
            n_jobs=n_jobs,
            resume=resume,
        )

    make_figure(ANY_PATH, FIG_PATH)


if __name__ == "__main__":
    main()
