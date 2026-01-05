# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **Unified API**: `vamos.optimize()` consolidates `study()` and `auto_optimize()` into a single powerful entry point.
- **Performance Profiler**: `vamos-profile` CLI for benchmarking engines (NumPy, Numba, JAX) and algorithms.
- **Interactive Dashboard**: `result.explore()` launches a Plotly-based interactive visualization of the Pareto front.
- **JAX Support**: New GPU-accelerated kernel backend (`engine="jax"`) for large-scale population updates.
- **New Algorithms**: Added `AGE-MOEA` (Adaptive Geometry Estimation) and `RVEA` (Reference Vector Guided EA).
- **Publication Tools**: `result.to_latex()` auto-generates LaTeX tables for research papers.
- **Constraint Benchmarks**: New test suite validating performance on constrained problems (CTP, OSY).
- **Distributed Examples**: Added Dask cluster example (`examples/distributed/dask_cluster.py`) and scaling docs.
- **AI Agent Context**: Added `AGENTS.md` files throughout the codebase to help AI coding assistants understand architecture and conventions.

### Changed
- **Dependency Hygiene**: Simplified `pyproject.toml` extras into 4 core groups: `compute`, `research`, `analysis`, `dev`.
- **Type Safety**: Improved type hints in fluent API (`StudyBuilder`) using `Self` types for better IDE support.
