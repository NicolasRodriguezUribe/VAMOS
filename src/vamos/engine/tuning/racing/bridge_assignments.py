"""Assignment -> config builders for tuning."""

from __future__ import annotations

from typing import Any, cast

from vamos.engine.algorithm.config.types import AlgorithmConfigProtocol

from ._bridge_assignment_builders import build_moead_permutation_config, config_builders
from ._bridge_assignment_shared import sanitize_assignment


def config_from_assignment(algorithm_name: str, assignment: dict[str, Any]) -> AlgorithmConfigProtocol:
    """Build a concrete algorithm config dataclass from a sampled assignment."""
    algo = algorithm_name.lower()
    assignment = sanitize_assignment(assignment)

    if algo == "moead_permutation":
        return build_moead_permutation_config(assignment)

    for base_name, (names, builder) in config_builders().items():
        if algo == f"{base_name}_mixed":
            return cast(AlgorithmConfigProtocol, builder(assignment, mixed=True))
        if algo in names:
            return cast(AlgorithmConfigProtocol, builder(assignment))

    raise ValueError(f"Unsupported algorithm for config construction: {algorithm_name}")


__all__ = ["config_from_assignment"]
