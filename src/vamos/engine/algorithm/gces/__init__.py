"""GCES algorithm family module."""

from .gces import (
    GCES,
    GCESNoComp,
    GCESNoGeo,
    NSGA2CurvGap,
    NSGA2Farthest,
    NSGA2GapFill,
    NSGA2HVFarthest,
    NSGA2HVRefFarthest,
    NSGA2RefCoverFarthest,
    NSGA2SectorFarthest,
)

__all__ = [
    "GCES",
    "GCESNoComp",
    "GCESNoGeo",
    "NSGA2Farthest",
    "NSGA2GapFill",
    "NSGA2CurvGap",
    "NSGA2HVFarthest",
    "NSGA2RefCoverFarthest",
    "NSGA2HVRefFarthest",
    "NSGA2SectorFarthest",
]
