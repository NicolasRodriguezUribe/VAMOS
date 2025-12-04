from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from vamos.problem.registry import (
    ProblemSelection,
    available_problem_names,
    make_problem_selection,
)

# Constants moved from runner.py
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Preset names map to problem keys defined in the registry.
PROBLEM_SET_PRESETS: dict[str, Sequence[str]] = {
    "families": ("zdt1", "dtlz2", "wfg4", "tsp6"),
    "zdt": ("zdt1", "zdt2", "zdt3", "zdt4", "zdt6"),
    "dtlz": ("dtlz1", "dtlz2", "dtlz3", "dtlz4"),
    "wfg": ("wfg1", "wfg2", "wfg3", "wfg4", "wfg5", "wfg6", "wfg7", "wfg8", "wfg9"),
    "tsp": ("tsp6",),
    "tsplib": ("kroa100", "krob100", "kroc100", "krod100", "kroe100"),
}

REFERENCE_FRONT_PATHS = {
    "zdt1": PROJECT_ROOT / "data/reference_fronts/ZDT1.csv",
    "zdt2": PROJECT_ROOT / "data/reference_fronts/ZDT2.csv",
    "zdt3": PROJECT_ROOT / "data/reference_fronts/ZDT3.csv",
    "zdt4": PROJECT_ROOT / "data/reference_fronts/ZDT4.csv",
    "zdt6": PROJECT_ROOT / "data/reference_fronts/ZDT6.csv",
}


def resolve_reference_front_path(problem_key: str, explicit_path: str | None) -> str | None:
    """
    Return a path to a reference front, using an explicit override when provided.
    """
    if explicit_path:
        return explicit_path
    if problem_key in REFERENCE_FRONT_PATHS:
        p = REFERENCE_FRONT_PATHS[problem_key]
        if p.exists():
            return str(p)
    return None


def _validate_problem_name(name: str) -> None:
    if name not in available_problem_names():
        available = ", ".join(sorted(available_problem_names()))
        raise ValueError(f"Unknown problem '{name}'. Available problems: {available}")


def _make_selection(name: str, n_var: int | None, n_obj: int | None) -> ProblemSelection:
    _validate_problem_name(name)
    return make_problem_selection(name, n_var=n_var, n_obj=n_obj)


def resolve_problem_selection(args) -> Iterable[ProblemSelection]:
    """
    Decide which problems to run based on CLI args, returning registry-backed selections.
    """
    if getattr(args, "problem_set", None):
        preset_name = args.problem_set
        if preset_name not in PROBLEM_SET_PRESETS:
            raise ValueError(f"Unknown problem set: {preset_name}")
        return tuple(_make_selection(name, args.n_var, args.n_obj) for name in PROBLEM_SET_PRESETS[preset_name])

    if getattr(args, "problem", None):
        return (_make_selection(args.problem, args.n_var, args.n_obj),)

    # Fallback or default
    return (_make_selection("zdt1", args.n_var, args.n_obj),)


def resolve_problem_selections(args) -> Iterable[ProblemSelection]:
    """
    Return a list of ProblemSelection objects to run.
    Handles 'all', specific sets, or single problems.
    """
    if getattr(args, "problem_set", None):
        if args.problem_set == "all":
            all_names = []
            for key, problems in PROBLEM_SET_PRESETS.items():
                if key == "families":
                    continue
                all_names.extend(problems)
            return tuple(_make_selection(name, args.n_var, args.n_obj) for name in all_names)
        return resolve_problem_selection(args)

    if getattr(args, "problem", None):
        return (_make_selection(args.problem, args.n_var, args.n_obj),)

    return (_make_selection("zdt1", args.n_var, args.n_obj),)
