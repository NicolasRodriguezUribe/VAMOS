# algorithm/variation_pipeline.py
"""
Backward-compatibility shim for variation_pipeline module.

The implementation has moved to vamos.engine.algorithm.components.variation.pipeline.
This module re-exports the public API for backward compatibility.
"""
from vamos.engine.algorithm.components.variation.pipeline import VariationPipeline

__all__ = ["VariationPipeline"]
