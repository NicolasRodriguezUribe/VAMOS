"""
Lightweight self-check to verify a VAMOS installation.

Runs tiny NSGA-II jobs on ZDT1 across available backends and reports results.
Intended for quick sanity checks; not a benchmark.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.problem.registry import make_problem_selection
from vamos.experiment.runner import run_single


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def _configure_cli_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
    root.setLevel(level)


@dataclass
class CheckResult:
    name: str
    status: str  # "ok", "skipped", "failed"
    detail: str | None = None


def _run_backend_check(engine: str, *, pop_size: int = 6, max_eval: int = 20) -> CheckResult:
    selection = make_problem_selection("zdt1", n_var=6)
    cfg = ExperimentConfig(population_size=pop_size, offspring_population_size=pop_size, max_evaluations=max_eval, seed=1)
    try:
        run_single(engine, "nsgaii", selection, cfg, selection_pressure=2)
        return CheckResult(name=f"nsgaii-{engine}", status="ok")
    except ImportError as exc:
        return CheckResult(name=f"nsgaii-{engine}", status="skipped", detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        return CheckResult(name=f"nsgaii-{engine}", status="failed", detail=str(exc))


def run_self_check(verbose: bool = False) -> list[CheckResult]:
    """Run a minimal set of smoke checks for each compute backend.

    Always exercises NumPy; Numba and MooCore are optional and reported
    as ``"skipped"`` when the dependency is missing.

    Parameters
    ----------
    verbose : bool, default False
        If ``True``, log the result of each check at INFO level.

    Returns
    -------
    list[CheckResult]
        One entry per backend / encoding variant with ``status`` in
        ``{"ok", "skipped", "failed"}``.

    Raises
    ------
    RuntimeError
        If the mandatory NumPy backend check fails.
    """
    checks: list[CheckResult] = []
    for engine in ("numpy", "numba", "moocore"):
        result = _run_backend_check(engine)
        checks.append(result)
        if verbose:
            status = result.status.upper()
            msg = result.detail or ""
            _logger().info("[self-check] %s: %s %s", result.name, status, msg)

    # Binary and mixed smoke on NumPy only
    for name, label in (("bin_knapsack", "binary"), ("mixed_design", "mixed")):
        selection = make_problem_selection(name)
        cfg = ExperimentConfig(population_size=8, offspring_population_size=8, max_evaluations=20, seed=2)
        try:
            run_single("numpy", "nsgaii", selection, cfg, selection_pressure=2)
            checks.append(CheckResult(name=f"{name}", status="ok"))
        except Exception as exc:  # pragma: no cover - quick smoke only
            checks.append(CheckResult(name=f"{name}", status="failed", detail=str(exc)))

    numpy_ok = next((c for c in checks if c.name == "nsgaii-numpy"), None)
    if numpy_ok is None or numpy_ok.status != "ok":
        detail = numpy_ok.detail if numpy_ok else "NumPy check missing"
        raise RuntimeError(f"VAMOS self-check failed for NumPy backend: {detail}")
    return checks


def main() -> None:
    """Entry-point for `python -m vamos.experiment.diagnostics.self_check`."""
    _configure_cli_logging()
    checks = run_self_check(verbose=True)
    failed = [c for c in checks if c.status == "failed"]
    skipped = [c for c in checks if c.status == "skipped"]
    if failed:
        raise SystemExit(1)
    if skipped:
        _logger().info("[self-check] Skipped: %s", ", ".join(c.name for c in skipped))
    _logger().info("[self-check] Completed.")


if __name__ == "__main__":
    main()
