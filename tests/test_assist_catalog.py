from __future__ import annotations

from vamos.assist.catalog import build_catalog


REQUIRED_KEYS: tuple[str, ...] = (
    "algorithms",
    "kernels",
    "crossover_methods",
    "mutation_methods",
    "templates",
)


def test_build_catalog_required_keys_and_types() -> None:
    catalog = build_catalog()
    for key in REQUIRED_KEYS:
        assert key in catalog
        assert isinstance(catalog[key], list)


def test_build_catalog_lists_are_sorted() -> None:
    catalog = build_catalog(problem_type="int")
    for key in REQUIRED_KEYS:
        values = catalog[key]
        assert values == sorted(values)


def test_build_catalog_has_algorithms_and_kernels() -> None:
    catalog = build_catalog()
    assert catalog["algorithms"]
    assert catalog["kernels"]
