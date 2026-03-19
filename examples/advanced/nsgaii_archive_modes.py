"""
NSGA-II archive-family modes on a tiny deterministic ZDT1 run.

Runs three configurations:
1) baseline NSGA-II (`archive_mode="off"`)
2) passive archive (`archive_mode="passive"`)
3) archive-aware split-front survival (`archive_mode="hybrid_survival"`)

The script prints compact archive and diagnostics information so the mode
behavior is easy to inspect without reading internal state.

Usage:
    python examples/advanced/nsgaii_archive_modes.py

Optional environment variables:
    VAMOS_ARCHIVE_MODES_ENGINE  (default: numpy)
    VAMOS_ARCHIVE_MODES_EVALS   (default: 120)
    VAMOS_ARCHIVE_MODES_POP     (default: 20)
    VAMOS_ARCHIVE_MODES_SEED    (default: 7)
"""

from __future__ import annotations

import os

from vamos import optimize
from vamos.algorithms import NSGAIIConfig
from vamos.problems import ZDT1


ENGINE = os.environ.get("VAMOS_ARCHIVE_MODES_ENGINE", "numpy")
MAX_EVALUATIONS = int(os.environ.get("VAMOS_ARCHIVE_MODES_EVALS", "120"))
POP_SIZE = int(os.environ.get("VAMOS_ARCHIVE_MODES_POP", "20"))
SEED = int(os.environ.get("VAMOS_ARCHIVE_MODES_SEED", "7"))


def build_config(archive_mode: str) -> NSGAIIConfig:
    builder = (
        NSGAIIConfig.builder()
        .pop_size(POP_SIZE)
        .offspring_size(POP_SIZE)
        .crossover("sbx", prob=0.9, eta=15.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
        .archive_mode(archive_mode)
    )
    if archive_mode == "hybrid_survival":
        builder = builder.archive_hybrid_alpha(0.5).archive_hybrid_k(3)
    return builder.build()


def summarize(label: str, archive_mode: str) -> None:
    result = optimize(
        ZDT1(n_var=12),
        algorithm="nsgaii",
        algorithm_config=build_config(archive_mode),
        termination=("max_evaluations", MAX_EVALUATIONS),
        seed=SEED,
        engine=ENGINE,
    )

    data = result.data
    archive = data.get("archive") or {}
    diagnostics = data.get("archive_diagnostics") or {}
    population = data.get("population") or {}

    population_size = len(population.get("F", []))
    archive_size = int(archive.get("size", 0))

    print(f"\n[{label}]")
    print(f"  returned_size={len(result)}")
    print(f"  population_size={population_size}")
    print(f"  archive_size={archive_size}")
    print(f"  execution_mode={diagnostics.get('execution_mode')}")
    print(f"  survival_path={diagnostics.get('survival_path')}")
    print(f"  hybrid_status={diagnostics.get('hybrid_status')}")
    print(f"  hybrid_fallback_reason={diagnostics.get('hybrid_fallback_reason')}")
    print(f"  hybrid_split_front_mode={diagnostics.get('hybrid_split_front_mode')}")
    print(f"  hybrid_split_front_reason={diagnostics.get('hybrid_split_front_reason')}")


def main() -> None:
    print("NSGA-II archive-family modes")
    print(f"  engine={ENGINE}, evaluations={MAX_EVALUATIONS}, pop_size={POP_SIZE}, seed={SEED}")

    summarize("standard", "off")
    summarize("passive_archive", "passive")
    summarize("hybrid_survival", "hybrid_survival")


if __name__ == "__main__":
    main()
