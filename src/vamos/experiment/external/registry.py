from __future__ import annotations

import logging

from vamos.foundation.problem.registry import ProblemSelection

from .jmetalpy import _run_jmetalpy_nsga2, _run_jmetalpy_perm_nsga2
from .pymoo import _run_pymoo_nsga2, _run_pymoo_perm_nsga2
from .pygmo import _run_pygmo_nsga2

logger = logging.getLogger(__name__)


class ExternalAlgorithmAdapter:
    """
    Thin adapter to standardize external baseline invocation.
    """

    def __init__(self, name: str, runner_fn):
        self.name = name
        self._runner_fn = runner_fn

    def run(
        self,
        selection: ProblemSelection,
        *,
        use_native_problem: bool,
        config,
        make_metrics,
        print_banner,
        print_results,
    ):
        return self._runner_fn(
            selection,
            use_native_problem=use_native_problem,
            config=config,
            make_metrics=make_metrics,
            print_banner=print_banner,
            print_results=print_results,
        )


EXTERNAL_ALGORITHM_RUNNERS = {
    "pymoo_nsga2": _run_pymoo_nsga2,
    "jmetalpy_nsga2": _run_jmetalpy_nsga2,
    "pygmo_nsga2": _run_pygmo_nsga2,
    "pymoo_perm_nsga2": _run_pymoo_perm_nsga2,
    "jmetalpy_perm_nsga2": _run_jmetalpy_perm_nsga2,
}

EXTERNAL_ALGORITHM_ADAPTERS = {name: ExternalAlgorithmAdapter(name, fn) for name, fn in EXTERNAL_ALGORITHM_RUNNERS.items()}


def resolve_external_algorithm(name: str) -> ExternalAlgorithmAdapter:
    try:
        return EXTERNAL_ALGORITHM_ADAPTERS[name]
    except KeyError as exc:  # pragma: no cover - defensive guard
        available = ", ".join(sorted(EXTERNAL_ALGORITHM_ADAPTERS))
        raise ValueError(f"Unknown external algorithm '{name}'. Available: {available}") from exc


def run_external(
    name: str,
    selection: ProblemSelection,
    *,
    use_native_problem: bool,
    config,
    make_metrics,
    print_banner,
    print_results,
):
    adapter = resolve_external_algorithm(name)
    try:
        return adapter.run(
            selection,
            use_native_problem=use_native_problem,
            config=config,
            make_metrics=make_metrics,
            print_banner=print_banner,
            print_results=print_results,
        )
    except ImportError as exc:
        logger.warning("Skipping %s: %s", name, exc)
        logger.info("%s", "=" * 80)
        return None
