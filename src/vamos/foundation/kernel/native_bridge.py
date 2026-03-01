from __future__ import annotations

import os
import shutil
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

_INSTALL_HINT = 'pip install -e ".[native]" or pip install vamospp'


class NativeNsga2Bridge:
    """Narrow adapter around the native ``vamospp`` module."""

    def __init__(self) -> None:
        self._module: Any | None = None

    @staticmethod
    def _bootstrap_windows_dll_paths() -> None:
        if os.name != "nt" or not hasattr(os, "add_dll_directory"):
            return

        candidates = [
            os.environ.get("VAMOSPP_DLL_DIR"),
            str(Path(sys.executable).resolve().parent),
        ]
        for tool in ("g++", "gcc", "clang++"):
            tool_path = shutil.which(tool)
            if tool_path:
                candidates.append(str(Path(tool_path).resolve().parent))

        seen: set[str] = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            path = Path(candidate)
            if not path.exists():
                continue
            try:
                os.add_dll_directory(str(path))
            except (FileNotFoundError, OSError):
                pass

    def _load(self) -> Any:
        if self._module is not None:
            return self._module

        self._bootstrap_windows_dll_paths()
        try:
            module = import_module("vamospp")
        except ImportError as exc:
            raise ImportError(
                "Kernel 'numba-mixed' requires the [native] extra (vamospp>=0.1). "
                f"Install with `{_INSTALL_HINT}`."
            ) from exc

        native = False
        is_native = getattr(module, "is_native_backend", None)
        if callable(is_native):
            native = bool(is_native())
        elif hasattr(module, "backend_info"):
            info_fn = getattr(module, "backend_info")
            if callable(info_fn):
                payload = info_fn()
                if isinstance(payload, dict):
                    native = bool(payload.get("native", False))

        if not native:
            raise ImportError(
                "Kernel 'numba-mixed' requires a native vamospp build. "
                f"Install with `{_INSTALL_HINT}` and rebuild the native extension."
            )

        self._module = module
        return module

    def require_native(self) -> None:
        self._load()

    def available(self) -> bool:
        try:
            self._load()
        except ImportError:
            return False
        return True

    def backend_info(self) -> dict[str, object]:
        module = self._load()
        info_fn = getattr(module, "backend_info", None)
        if callable(info_fn):
            payload = info_fn()
            if isinstance(payload, dict):
                return payload
        return {"backend": "unknown", "native": True}

    @staticmethod
    def _as_float64_c(value: np.ndarray, *, name: str, ndim: int) -> np.ndarray:
        arr = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
        if arr.ndim != ndim:
            raise ValueError(f"{name} must be {ndim}D.")
        return arr

    def fast_non_dominated_sort(self, F: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
        module = self._load()
        F_arr = self._as_float64_c(F, name="F", ndim=2)
        fronts, ranks = module.fast_non_dominated_sort(F_arr)
        return list(fronts), np.ascontiguousarray(np.asarray(ranks, dtype=np.int64))

    def crowding_distance(self, F: np.ndarray, fronts: list[list[int]] | None) -> np.ndarray:
        module = self._load()
        F_arr = self._as_float64_c(F, name="F", ndim=2)
        crowd = module.crowding_distance(F_arr, fronts)
        return np.ascontiguousarray(np.asarray(crowd, dtype=np.float64))

    def nsga2_survival(
        self,
        X: np.ndarray,
        F: np.ndarray,
        X_off: np.ndarray,
        F_off: np.ndarray,
        pop_size: int,
        *,
        return_indices: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        module = self._load()
        payload = module.nsga2_survival(
            self._as_float64_c(X, name="X", ndim=2),
            self._as_float64_c(F, name="F", ndim=2),
            self._as_float64_c(X_off, name="X_off", ndim=2),
            self._as_float64_c(F_off, name="F_off", ndim=2),
            int(pop_size),
            bool(return_indices),
        )
        if return_indices:
            X_new, F_new, indices = payload
            return (
                self._as_float64_c(X_new, name="nsga2_survival X", ndim=2),
                self._as_float64_c(F_new, name="nsga2_survival F", ndim=2),
                np.ascontiguousarray(np.asarray(indices, dtype=np.int64)),
            )
        X_new, F_new = payload
        return (
            self._as_float64_c(X_new, name="nsga2_survival X", ndim=2),
            self._as_float64_c(F_new, name="nsga2_survival F", ndim=2),
        )


__all__ = ["NativeNsga2Bridge"]
