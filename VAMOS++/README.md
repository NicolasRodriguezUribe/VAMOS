# VAMOS++

`vamospp` is the native acceleration package for VAMOS.

It provides:

- A nanobind C++ extension module (`vamospp._core`)
- A Python fallback implementation (`vamospp._fallback`) with the same API
- A stable import surface (`import vamospp`) used by the VAMOS `cpp` kernel

The fallback keeps functionality available when native wheels are unavailable,
while the C++ extension can progressively replace hot kernels.
