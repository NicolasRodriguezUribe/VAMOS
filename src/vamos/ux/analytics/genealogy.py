from __future__ import annotations

from typing import Any

from vamos.engine.hooks.genealogy import (
    DefaultGenealogyTracker as GenealogyTracker,
)
from vamos.engine.hooks.genealogy import (
    GenealogyRecord,
    IndividualID,
    get_lineage,
)
from vamos.engine.hooks.genealogy import (
    generation_contributions as _generation_contributions,
)
from vamos.engine.hooks.genealogy import (
    operator_success_stats as _operator_success_stats,
)


def compute_operator_success_stats(tracker: GenealogyTracker) -> Any:
    import pandas as pd

    return pd.DataFrame(_operator_success_stats(tracker))


def compute_generation_contributions(tracker: GenealogyTracker) -> Any:
    import pandas as pd

    return pd.DataFrame(_generation_contributions(tracker))


__all__ = [
    "IndividualID",
    "GenealogyRecord",
    "GenealogyTracker",
    "get_lineage",
    "compute_operator_success_stats",
    "compute_generation_contributions",
]
