import importlib
import types

import pytest

from vamos.foundation.kernel.registry import resolve_kernel


def test_unknown_kernel_errors():
    with pytest.raises(ValueError, match="Unknown engine|Unknown kernel"):
        resolve_kernel("does_not_exist")


def test_numba_missing_dependency(monkeypatch):
    def fake_import(name, *args, **kwargs):
        if name.endswith("numba_backend"):
            raise ImportError("forced-missing-numba")
        return importlib.import_module(name, package=None)

    monkeypatch.setattr("vamos.foundation.kernel.registry.import_module", fake_import)
    with pytest.raises(ImportError, match="requires the \\[backends\\] extra"):
        resolve_kernel("numba")


def test_moocore_missing_dependency(monkeypatch):
    def fake_import(name, *args, **kwargs):
        if name.endswith("moocore_backend"):
            raise ImportError("forced-missing-moocore")
        return importlib.import_module(name, package=None)

    monkeypatch.setattr("vamos.foundation.kernel.registry.import_module", fake_import)
    with pytest.raises(ImportError, match="requires the \\[backends\\] extra"):
        resolve_kernel("moocore")
