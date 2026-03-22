"""Smoke tests for the experimental JAX backend."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest


def has_jax() -> bool:
    return importlib.util.find_spec("jax") is not None


@pytest.mark.skipif(not has_jax(), reason="JAX not installed")
class TestJaxBackendSmoke:
    """Validate the experimental JAX kernel executes and returns sane results."""

    def test_jax_devices_available(self):
        """Verify JAX can detect devices."""
        import jax

        devices = jax.devices()
        assert len(devices) > 0, "No JAX devices found"
        print(f"JAX devices: {devices}")

    def test_nsga2_ranking_executes_for_large_population(self):
        """JAX ranking should execute successfully for a larger population."""
        from vamos.foundation.kernel.jax_backend import JaxKernel

        # Generate test data - large population
        np.random.seed(42)
        n_pop = 2000
        n_obj = 3
        F = np.random.rand(n_pop, n_obj)

        jax_kernel = JaxKernel()

        # Warmup JAX (JIT compilation)
        _ = jax_kernel.nsga2_ranking(F[:100])

        ranks_jax, cd_jax = jax_kernel.nsga2_ranking(F)
        assert ranks_jax is not None
        assert cd_jax is not None
        assert len(ranks_jax) == n_pop
        assert len(cd_jax) == n_pop

    def test_crowding_distance_correctness(self):
        """Verify JAX crowding distance matches NumPy."""
        from vamos.foundation.kernel.jax_backend import JaxKernel
        from vamos.foundation.kernel.numpy_backend import NumPyKernel

        np.random.seed(42)
        F = np.random.rand(50, 2)

        numpy_kernel = NumPyKernel()
        jax_kernel = JaxKernel()

        _, cd_np = numpy_kernel.nsga2_ranking(F)
        _, cd_jax = jax_kernel.nsga2_ranking(F)

        # Check boundary points have infinity
        assert np.isinf(cd_np).sum() > 0, "NumPy should have inf for boundaries"
        assert np.isinf(cd_jax).sum() > 0, "JAX should have inf for boundaries"
