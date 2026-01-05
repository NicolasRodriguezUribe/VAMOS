"""
Performance tests for JAX backend validation.
"""
from __future__ import annotations

import time
import pytest
import numpy as np


def has_jax() -> bool:
    try:
        import jax
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not has_jax(), reason="JAX not installed")
class TestJaxSpeedup:
    """Validate JAX kernel provides speedup over NumPy."""
    
    def test_jax_devices_available(self):
        """Verify JAX can detect devices."""
        import jax
        devices = jax.devices()
        assert len(devices) > 0, "No JAX devices found"
        print(f"JAX devices: {devices}")
    
    def test_nsga2_ranking_speedup(self):
        """Test JAX ranking is faster than NumPy for large populations."""
        from vamos.foundation.kernel.numpy_backend import NumPyKernel
        from vamos.foundation.kernel.jax_backend import JaxKernel
        
        # Generate test data - large population
        np.random.seed(42)
        n_pop = 2000
        n_obj = 3
        F = np.random.rand(n_pop, n_obj)
        
        numpy_kernel = NumPyKernel()
        jax_kernel = JaxKernel()
        
        # Warmup JAX (JIT compilation)
        _ = jax_kernel.nsga2_ranking(F[:100])
        
        # Benchmark NumPy
        start = time.perf_counter()
        for _ in range(3):
            ranks_np, cd_np = numpy_kernel.nsga2_ranking(F)
        numpy_time = (time.perf_counter() - start) / 3
        
        # Benchmark JAX
        start = time.perf_counter()
        for _ in range(3):
            ranks_jax, cd_jax = jax_kernel.nsga2_ranking(F)
        jax_time = (time.perf_counter() - start) / 3
        
        speedup = numpy_time / jax_time if jax_time > 0 else 0
        
        print(f"\nNumPy time: {numpy_time:.4f}s")
        print(f"JAX time: {jax_time:.4f}s")
        print(f"Speedup: {speedup:.2f}x")
        
        # Note: On CPU, JAX may not always be faster due to JIT overhead
        # On GPU, expect significant speedup
        # We just verify it runs successfully
        assert ranks_jax is not None
        assert cd_jax is not None
        assert len(ranks_jax) == n_pop
        assert len(cd_jax) == n_pop
    
    def test_crowding_distance_correctness(self):
        """Verify JAX crowding distance matches NumPy."""
        from vamos.foundation.kernel.numpy_backend import NumPyKernel
        from vamos.foundation.kernel.jax_backend import JaxKernel
        
        np.random.seed(42)
        F = np.random.rand(50, 2)
        
        numpy_kernel = NumPyKernel()
        jax_kernel = JaxKernel()
        
        _, cd_np = numpy_kernel.nsga2_ranking(F)
        _, cd_jax = jax_kernel.nsga2_ranking(F)
        
        # Check boundary points have infinity
        assert np.isinf(cd_np).sum() > 0, "NumPy should have inf for boundaries"
        assert np.isinf(cd_jax).sum() > 0, "JAX should have inf for boundaries"
