from __future__ import annotations

ALGORITHM_VARIANT_GROUPS: dict[str, frozenset[str]] = {
    "nsgaii": frozenset({"nsgaii", "nsgaii_permutation", "nsgaii_mixed", "nsgaii_binary", "nsgaii_integer"}),
    "moead": frozenset({"moead", "moead_permutation", "moead_mixed", "moead_binary", "moead_integer"}),
    "nsgaiii": frozenset({"nsgaiii", "nsgaiii_permutation", "nsgaiii_mixed", "nsgaiii_binary", "nsgaiii_integer"}),
    "smsemoa": frozenset({"smsemoa", "smsemoa_permutation", "smsemoa_mixed", "smsemoa_binary", "smsemoa_integer"}),
    "spea2": frozenset({"spea2", "spea2_permutation", "spea2_mixed", "spea2_binary", "spea2_integer"}),
    "ibea": frozenset({"ibea", "ibea_permutation", "ibea_mixed", "ibea_binary", "ibea_integer"}),
    "smpso": frozenset({"smpso", "smpso_mixed"}),
    "agemoea": frozenset({"agemoea", "agemoea_permutation", "agemoea_mixed", "agemoea_binary", "agemoea_integer"}),
    "rvea": frozenset({"rvea", "rvea_permutation", "rvea_mixed", "rvea_binary", "rvea_integer"}),
}


VARIANT_TO_CANONICAL: dict[str, str] = {
    variant: canonical for canonical, variants in ALGORITHM_VARIANT_GROUPS.items() for variant in variants
}


PERMUTATION_COMPATIBLE_ALGORITHMS: frozenset[str] = frozenset(
    canonical
    for canonical, variants in ALGORITHM_VARIANT_GROUPS.items()
    if any(name.endswith("_permutation") for name in variants)
)


def canonical_algorithm_name(name: str) -> str:
    return VARIANT_TO_CANONICAL.get(str(name).lower(), str(name).lower())


__all__ = [
    "ALGORITHM_VARIANT_GROUPS",
    "VARIANT_TO_CANONICAL",
    "PERMUTATION_COMPATIBLE_ALGORITHMS",
    "canonical_algorithm_name",
]
