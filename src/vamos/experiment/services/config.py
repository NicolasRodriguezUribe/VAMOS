from __future__ import annotations

from vamos.engine.config.variation import VariationConfig, normalize_variation_config


def normalize_variations(
    *,
    nsgaii_variation: VariationConfig | None = None,
    moead_variation: VariationConfig | None = None,
    smsemoa_variation: VariationConfig | None = None,
    nsgaiii_variation: VariationConfig | None = None,
    spea2_variation: VariationConfig | None = None,
    ibea_variation: VariationConfig | None = None,
    smpso_variation: VariationConfig | None = None,
) -> tuple[
    VariationConfig | None,
    VariationConfig | None,
    VariationConfig | None,
    VariationConfig | None,
    VariationConfig | None,
    VariationConfig | None,
    VariationConfig | None,
]:
    return (
        normalize_variation_config(nsgaii_variation),
        normalize_variation_config(moead_variation),
        normalize_variation_config(smsemoa_variation),
        normalize_variation_config(nsgaiii_variation),
        normalize_variation_config(spea2_variation),
        normalize_variation_config(ibea_variation),
        normalize_variation_config(smpso_variation),
    )


__all__ = ["VariationConfig", "normalize_variation_config", "normalize_variations"]
