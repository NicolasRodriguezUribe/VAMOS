from __future__ import annotations

import pytest

from vamos import optimize
from vamos.foundation.kernel.registry import resolve_kernel


def test_kernel_profile_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VAMOS_PROFILE_KERNELS", raising=False)
    result = optimize(
        "zdt1",
        algorithm="nsgaii",
        engine="numpy",
        max_evaluations=400,
        pop_size=20,
        seed=0,
    )
    assert "kernel_profile" not in result.data


@pytest.mark.parametrize("algorithm", ["nsgaii", "smsemoa", "spea2"])
def test_kernel_profile_enabled_via_env(monkeypatch: pytest.MonkeyPatch, algorithm: str) -> None:
    monkeypatch.setenv("VAMOS_PROFILE_KERNELS", "1")
    result = optimize(
        "zdt1",
        algorithm=algorithm,
        engine="numpy",
        max_evaluations=400,
        pop_size=20,
        seed=0,
    )
    profile = result.data.get("kernel_profile")
    assert isinstance(profile, dict)
    assert profile.get("enabled") is True
    assert int(profile.get("n_generations", 0)) > 0
    per_kernel = profile.get("per_kernel", {})
    assert isinstance(per_kernel, dict)
    assert "generation_total" in per_kernel


def test_nsgaii_cpp_profile_uses_fused_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("vamospp")
    try:
        resolve_kernel("cpp")
    except ImportError:
        pytest.skip("cpp kernel is not available")

    monkeypatch.setenv("VAMOS_PROFILE_KERNELS", "1")
    result = optimize(
        "zdt1",
        algorithm="nsgaii",
        engine="cpp",
        max_evaluations=400,
        pop_size=20,
        seed=0,
    )
    profile = result.data.get("kernel_profile", {})
    per_kernel = profile.get("per_kernel", {})
    assert "generate_offspring" in per_kernel


@pytest.mark.parametrize("algorithm", ["smsemoa", "spea2"])
def test_cpp_profile_uses_fused_generation_for_supported_algorithms(monkeypatch: pytest.MonkeyPatch, algorithm: str) -> None:
    pytest.importorskip("vamospp")
    try:
        resolve_kernel("cpp")
    except ImportError:
        pytest.skip("cpp kernel is not available")

    monkeypatch.setenv("VAMOS_PROFILE_KERNELS", "1")
    result = optimize(
        "zdt1",
        algorithm=algorithm,
        engine="cpp",
        max_evaluations=400,
        pop_size=20,
        seed=0,
    )
    profile = result.data.get("kernel_profile", {})
    per_kernel = profile.get("per_kernel", {})
    assert "generate_offspring" in per_kernel
