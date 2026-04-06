"""Thin compatibility layer for algorithm builder entrypoints."""

from __future__ import annotations

from vamos.engine.algorithm._builder_adaptive import (
    build_agemoea_algorithm,
    build_smpso_algorithm,
)
from vamos.engine.algorithm._builder_archive_population import (
    build_gces_algorithm,
    build_gces_nocomp_algorithm,
    build_gces_nogeo_algorithm,
    build_ibea_algorithm,
    build_nsga2_curvgap_algorithm,
    build_nsga2_farthest_algorithm,
    build_nsga2_gapfill_algorithm,
    build_nsga2_hvfarthest_algorithm,
    build_nsga2_hvref_farthest_algorithm,
    build_nsga2_refcover_farthest_algorithm,
    build_nsga2_sector_farthest_algorithm,
    build_nsgaii_algorithm,
    build_smsemoa_algorithm,
    build_spea2_algorithm,
)
from vamos.engine.algorithm._builder_decomposition import (
    build_moead_algorithm,
    build_nsgaiii_algorithm,
    build_rvea_algorithm,
)

__all__ = [
    "build_gces_algorithm",
    "build_gces_nocomp_algorithm",
    "build_gces_nogeo_algorithm",
    "build_nsga2_curvgap_algorithm",
    "build_nsga2_farthest_algorithm",
    "build_nsga2_gapfill_algorithm",
    "build_nsga2_hvfarthest_algorithm",
    "build_nsga2_refcover_farthest_algorithm",
    "build_nsga2_hvref_farthest_algorithm",
    "build_nsga2_sector_farthest_algorithm",
    "build_nsgaii_algorithm",
    "build_moead_algorithm",
    "build_smsemoa_algorithm",
    "build_nsgaiii_algorithm",
    "build_spea2_algorithm",
    "build_ibea_algorithm",
    "build_smpso_algorithm",
    "build_agemoea_algorithm",
    "build_rvea_algorithm",
]
