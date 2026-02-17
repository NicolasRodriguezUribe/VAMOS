"""
AOS Showcase — IDE play-button runner
======================================
Edit the CONFIG block below, then hit Run / F5.
No environment variables or CLI arguments needed.

Output
------
  experiments/mic/mic_showcase.csv          — final HV per run
  experiments/mic/mic_showcase_anytime.csv  — HV at each checkpoint
  paper/mic/figures/showcase_convergence.pdf — convergence-curve figure
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Configuration — edit here ────────────────────────────────────────────────

N_EVALS  = 100_000   # function evaluations per run
N_SEEDS  = 30        # independent runs per (variant, problem)
N_JOBS   = -1        # parallel workers; -1 = all CPUs, 1 = sequential
ENGINE   = "numba"   # "numba" (fast) or "numpy" (portable)
RESUME   = False     # True  → append to existing CSV and skip done runs
                     # False → overwrite everything and start fresh

# Which variants to compare. Options: "baseline", "random", "aos", "aos_v2"
# "aos_v2" is the phase-aware version introduced in this session.
VARIANTS = ["baseline", "random", "aos", "aos_v2"]

# Which problems to run. The default 6 are the highest-impact AOS wins.
# Feel free to swap in any problem from the comprehensive suite:
#   UF:      cec2009_uf1..cec2009_uf10
#   LSMOP:   lsmop1..lsmop9
#   C-DTLZ:  c1dtlz1, c1dtlz3, c2dtlz2
#   DC-DTLZ: dc1dtlz1, dc1dtlz3, dc2dtlz1, dc2dtlz3
#   MW:      mw1, mw2, mw3, mw5, mw6, mw7
PROBLEMS = [
    "lsmop1",        # +406 % vs baseline (large-scale, 100 vars)
    "lsmop8",        # baseline HV ≈ 0; AOS achieves 0.252
    "dc1dtlz3",      # +376 % (discontinuously constrained)
    "cec2009_uf7",   # +17 %  (curved Pareto set)
    "mw5",           # +73 %  (constrained)
    "cec2009_uf1",   # curved Pareto set, AOS converges fast
]

# ── End of configuration ─────────────────────────────────────────────────────

ROOT_DIR    = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT_DIR / "paper" / "mic" / "scripts"

sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "paper"))

# Tell the experiment module which suite to load (needed at import time)
os.environ["VAMOS_MIC_SUITE"] = "comprehensive"

# Translate CONFIG into the env vars that the showcase script reads
os.environ["VAMOS_MIC_N_EVALS"]  = str(N_EVALS)
os.environ["VAMOS_MIC_N_SEEDS"]  = str(N_SEEDS)
os.environ["VAMOS_MIC_N_JOBS"]   = str(N_JOBS)
os.environ["VAMOS_MIC_ENGINE"]   = ENGINE
os.environ["VAMOS_MIC_RESUME"]   = "1" if RESUME else "0"

import importlib.util as _ilu

def _load(path: Path):
    spec = _ilu.spec_from_file_location(path.stem, path)
    mod  = _ilu.module_from_spec(spec)   # type: ignore[arg-type]
    spec.loader.exec_module(mod)          # type: ignore[union-attr]
    return mod

_showcase = _load(SCRIPTS_DIR / "04_run_showcase_experiment.py")

# Override the problem and variant lists with what was set above
_showcase.SHOWCASE_PROBLEMS[:] = PROBLEMS
_showcase.VARIANTS[:] = VARIANTS


if __name__ == "__main__":
    print(f"Variants : {VARIANTS}")
    print(f"Problems : {PROBLEMS}")
    print(f"Budget   : {N_EVALS:,} evals × {N_SEEDS} seeds")
    print(f"Engine   : {ENGINE}   Workers: {N_JOBS}")
    print()

    _showcase.run_experiment(
        n_evals=N_EVALS,
        n_seeds=N_SEEDS,
        engine=ENGINE,
        n_jobs=N_JOBS if N_JOBS > 0 else max(1, (__import__("os").cpu_count() or 2) - 1),
        resume=RESUME,
    )

    _showcase.make_figure(_showcase.ANY_PATH, _showcase.FIG_PATH)
    print(f"\nDone. Figure: {_showcase.FIG_PATH}")
