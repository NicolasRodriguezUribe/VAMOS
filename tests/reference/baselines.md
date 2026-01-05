# Reference Verification Baselines

These tests verify **scientific correctness** by running algorithms on standard benchmarks with deterministic seeds.

## ZDT1 (Convex, 2D)
*   **Problem**: `ZDT1Problem(n_var=30)`
*   **Goal**: Minimize f1, f2
*   **Budget**: 2,500 evaluations
*   **Metric**: Hypervolume (Ref Point: [1.1, 1.1])
*   **Baselines**:
    *   **NSGA-II**: Expected HV > 0.60
    *   **MOEA/D**: Expected HV > 0.60

## ZDT2 (Non-Convex, 2D)
*   **Problem**: `ZDT2Problem(n_var=30)`
*   **Budget**: 3,000 evaluations
*   **Baselines**:
    *   **NSGA-II**: Expected HV > 0.30

## DTLZ2 (Concave, 3D)
*   **Problem**: `DTLZ2Problem(n_var=12, n_obj=3)`
*   **Budget**: 5,000 evaluations
*   **Metric**: Hypervolume (Ref Point: [1.1, 1.1, 1.1])
*   **Baselines**:
    *   **NSGA-III**: Expected HV > 0.50

> **Note**: These thresholds are lower bounds to catch significant regressions (e.g., algorithm collapses or fails to progress). They do not represent state-of-the-art performance, but rather "acceptable functioning" for regression testing.
