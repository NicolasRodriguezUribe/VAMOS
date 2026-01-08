from __future__ import annotations

from vamos.engine.config.variation import (
    merge_variation_overrides,
    normalize_operator_tuple,
    normalize_variation_config,
    resolve_default_variation_config,
)


def test_normalize_operator_tuple_accepts_str_tuple_dict():
    assert normalize_operator_tuple("sbx") == ("sbx", {})
    assert normalize_operator_tuple(("pm", {"prob": 0.1})) == ("pm", {"prob": 0.1})
    assert normalize_operator_tuple({"method": "uniform", "prob": 0.5}) == ("uniform", {"prob": 0.5})
    assert normalize_operator_tuple({"name": "repair", "alpha": 0.2}) == ("repair", {"alpha": 0.2})


def test_normalize_variation_config_handles_ops_and_extras():
    raw = {
        "crossover": {"method": "sbx", "prob": 0.9},
        "mutation": {"name": "pm", "eta": 20.0},
        "kappa": 0.05,
    }
    cfg = normalize_variation_config(raw)
    assert cfg == {
        "crossover": ("sbx", {"prob": 0.9}),
        "mutation": ("pm", {"eta": 20.0}),
        "kappa": 0.05,
    }


def test_merge_variation_overrides_prefers_non_null_override():
    base = {"crossover": ("sbx", {"prob": 0.9}), "kappa": 0.05}
    override = {"kappa": 0.1, "repair": ("clip", {})}
    merged = merge_variation_overrides(base, override)
    assert merged["crossover"] == base["crossover"]
    assert merged["kappa"] == 0.1
    assert merged["repair"] == ("clip", {})


def test_resolve_default_variation_config_defaults_by_encoding():
    real_cfg = resolve_default_variation_config("real", None)
    assert real_cfg["crossover"][0] == "sbx"
    assert real_cfg["mutation"][0] == "pm"

    binary_cfg = resolve_default_variation_config("binary", None)
    assert binary_cfg["crossover"][0] == "hux"
    assert binary_cfg["mutation"][0] == "bitflip"

    perm_cfg = resolve_default_variation_config("permutation", {"repair": ("clip", {})})
    assert perm_cfg["crossover"][0] == "ox"
    assert perm_cfg["repair"] == ("clip", {})
