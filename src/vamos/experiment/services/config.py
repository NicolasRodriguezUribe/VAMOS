from __future__ import annotations

from typing import Any

from vamos.engine.config.variation import normalize_variation_config

VariationConfig = dict[str, Any]


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
