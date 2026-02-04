"""
JAX GPU Acceleration Demo.

Demonstrates how to use the 'jax' engine for massive population sizes.
Requires `jax` and `jaxlib` installed (and a GPU for best performance).

Usage:
    python examples/jax_gpu_demo.py
"""

from __future__ import annotations
import time
from vamos import optimize


def run_numpy(pop_size=1000):
    print(f"\n--- NumPy (CPU) | Pop: {pop_size} ---")
    start = time.time()
    result = optimize("zdt1", algorithm="nsgaii", max_evaluations=10000, pop_size=pop_size, engine="numpy", verbose=False)
    elapsed = time.time() - start
    print(f"Time: {elapsed:.2f}s | Solutions: {len(result)}")
    return elapsed


def run_jax(pop_size=1000):
    print(f"\n--- JAX (GPU/Acc) | Pop: {pop_size} ---")
    try:
        import jax

        print(f"JAX devices: {jax.devices()}")
    except ImportError:
        print("JAX not installed. Skipping.")
        return 0

    start = time.time()
    result = optimize("zdt1", algorithm="nsgaii", max_evaluations=10000, pop_size=pop_size, engine="jax", verbose=False)
    elapsed = time.time() - start
    print(f"Time: {elapsed:.2f}s | Solutions: {len(result)}")
    return elapsed


def main():
    print("Comparing NumPy vs JAX for large populations...")

    # Small scale warm-up
    run_numpy(500)
    run_jax(500)

    # Larger scale (where GPU shines)
    # Note: Increase pop_size to 10k+ to see JAX win significantly
    t_np = run_numpy(2000)
    t_jax = run_jax(2000)

    if t_jax > 0:
        print(f"\nSpeedup: {t_np / t_jax:.2f}x")


if __name__ == "__main__":
    main()
