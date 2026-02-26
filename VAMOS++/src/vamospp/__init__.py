from __future__ import annotations

from . import _fallback

try:
    from . import _core as _native
except Exception:  # pragma: no cover - depends on build environment
    _native = None


def _resolve(name: str):
    if _native is not None and hasattr(_native, name):
        return getattr(_native, name)
    return getattr(_fallback, name)


def is_native_backend() -> bool:
    if _native is None:
        return False
    fn = getattr(_native, "is_native_backend", None)
    if callable(fn):
        return bool(fn())
    return True


def backend_info() -> dict[str, object]:
    info_fn = getattr(_native, "backend_info", None) if _native is not None else None
    if callable(info_fn):
        payload = info_fn()
        if isinstance(payload, dict):
            return payload
    return {"backend": "python-fallback", "native": False}


fast_non_dominated_sort = _resolve("fast_non_dominated_sort")
crowding_distance = _resolve("crowding_distance")
nsga2_ranking = _resolve("nsga2_ranking")
tournament_selection = _resolve("tournament_selection")
nsga2_survival = _resolve("nsga2_survival")
sbx_crossover = _resolve("sbx_crossover")
polynomial_mutation = _resolve("polynomial_mutation")
hypervolume = _resolve("hypervolume")
hypervolume_contributions = _resolve("hypervolume_contributions")
smsemoa_remove_index = _resolve("smsemoa_remove_index")
dominance_matrix = _resolve("dominance_matrix")
spea2_fitness = _resolve("spea2_fitness")
spea2_environmental_selection_indices = _resolve("spea2_environmental_selection_indices")
spea2_generate_offspring = _resolve("spea2_generate_offspring")
ibea_indicator_matrix = _resolve("ibea_indicator_matrix")
ibea_environmental_selection_indices = _resolve("ibea_environmental_selection_indices")
smsemoa_generate_offspring = _resolve("smsemoa_generate_offspring")
generate_offspring = _resolve("generate_offspring")
nsga2_evolve = _resolve("nsga2_evolve")


__all__ = [
    "backend_info",
    "crowding_distance",
    "dominance_matrix",
    "fast_non_dominated_sort",
    "generate_offspring",
    "hypervolume",
    "hypervolume_contributions",
    "ibea_environmental_selection_indices",
    "ibea_indicator_matrix",
    "is_native_backend",
    "nsga2_evolve",
    "nsga2_ranking",
    "nsga2_survival",
    "polynomial_mutation",
    "sbx_crossover",
    "smsemoa_generate_offspring",
    "smsemoa_remove_index",
    "spea2_fitness",
    "spea2_generate_offspring",
    "spea2_environmental_selection_indices",
    "tournament_selection",
]
