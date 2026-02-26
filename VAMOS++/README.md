# VAMOS++

`vamospp` is the native acceleration package for VAMOS.

It provides:

- A nanobind C++ extension module (`vamospp._core`)
- A Python fallback implementation (`vamospp._fallback`) with the same API
- A stable import surface (`import vamospp`) used by the VAMOS `cpp` kernel

The fallback keeps functionality available when native wheels are unavailable,
while the C++ extension can progressively replace hot kernels.

## NSGA-II native return contract

For the native path (`vamospp._core`), these functions return NumPy ndarrays
directly (C-contiguous `float64` / `int64`), not Python nested lists:

- `generate_offspring`: `ndarray[float64]` with shape `(n_offspring, n_var)`
- `nsga2_survival`: tuple `(X_new, F_new)` or `(X_new, F_new, indices)`
- `nsga2_evolve`: tuple `(X_new, F_new)`
- `smsemoa_generate_offspring`: `ndarray[float64]` with shape `(1, n_var)`
- `spea2_generate_offspring`: `ndarray[float64]` with shape `(n_offspring, n_var)`
