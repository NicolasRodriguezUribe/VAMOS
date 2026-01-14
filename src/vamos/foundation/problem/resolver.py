from __future__ import annotations

from argparse import Namespace
from difflib import get_close_matches
from collections.abc import Iterable, Sequence

from importlib.resources import as_file

from vamos.foundation.data import reference_front_path
from vamos.foundation.problem.registry import (
    ProblemSelection,
    available_problem_names,
    make_problem_selection,
)

# Preset names map to problem keys defined in the registry.
PROBLEM_SET_PRESETS: dict[str, Sequence[str]] = {
    "families": ("zdt1", "dtlz2", "wfg4", "tsp6"),
    "zdt": ("zdt1", "zdt2", "zdt3", "zdt4", "zdt6"),
    "dtlz": ("dtlz1", "dtlz2", "dtlz3", "dtlz4"),
    "wfg": ("wfg1", "wfg2", "wfg3", "wfg4", "wfg5", "wfg6", "wfg7", "wfg8", "wfg9"),
    "lz": ("lz09_f1", "lz09_f2", "lz09_f3", "lz09_f4", "lz09_f5", "lz09_f6", "lz09_f7", "lz09_f8", "lz09_f9"),
    "cec": ("cec2009_uf1", "cec2009_uf2", "cec2009_uf3", "cec2009_cf1"),
    "tsp": ("tsp6",),
    "tsplib": ("kroa100", "krob100", "kroc100", "krod100", "kroe100"),
    "real_world": ("ml_tuning", "welded_beam", "fs_real"),
}
_PROBLEM_DOCS = "docs/reference/problems.md"
_TROUBLESHOOTING_DOCS = "docs/guide/troubleshooting.md"


def _suggest_names(name: str, options: Sequence[str]) -> list[str]:
    if not name or not options:
        return []
    lookup = {option.lower(): option for option in options}
    matches = get_close_matches(name.lower(), lookup.keys(), n=3, cutoff=0.6)
    return [lookup[match] for match in matches]


def _format_unknown_problem(name: str, options: list[str]) -> str:
    parts = [f"Unknown problem '{name}'.", f"Available: {', '.join(options)}."]
    suggestions = _suggest_names(name, options)
    if suggestions:
        if len(suggestions) == 1:
            parts.append(f"Did you mean '{suggestions[0]}'?")
        else:
            parts.append("Did you mean one of: " + ", ".join(f"'{item}'" for item in suggestions) + "?")
    parts.append(f"Docs: {_PROBLEM_DOCS}.")
    parts.append(f"Troubleshooting: {_TROUBLESHOOTING_DOCS}.")
    return " ".join(parts)


def _format_unknown_problem_set(name: str, options: list[str]) -> str:
    parts = [f"Unknown problem set '{name}'.", f"Available: {', '.join(options)}."]
    suggestions = _suggest_names(name, options)
    if suggestions:
        if len(suggestions) == 1:
            parts.append(f"Did you mean '{suggestions[0]}'?")
        else:
            parts.append("Did you mean one of: " + ", ".join(f"'{item}'" for item in suggestions) + "?")
    parts.append(f"Docs: {_PROBLEM_DOCS}.")
    parts.append(f"Troubleshooting: {_TROUBLESHOOTING_DOCS}.")
    return " ".join(parts)


def resolve_reference_front_path(problem_key: str, explicit_path: str | None) -> str | None:
    """
    Return a path to a reference front, using an explicit override when provided.
    """
    if explicit_path:
        return explicit_path
    try:
        with as_file(reference_front_path(problem_key)) as p:
            return str(p)
    except Exception:
        return None


def _validate_problem_name(name: str) -> None:
    available = sorted(available_problem_names())
    if name not in available:
        raise ValueError(_format_unknown_problem(name, available))


def _make_selection(name: str, n_var: int | None, n_obj: int | None) -> ProblemSelection:
    _validate_problem_name(name)
    return make_problem_selection(name, n_var=n_var, n_obj=n_obj)


def resolve_problem_selection(args: Namespace) -> Iterable[ProblemSelection]:
    """
    Decide which problems to run based on CLI args, returning registry-backed selections.
    """
    if getattr(args, "problem_set", None):
        preset_name = args.problem_set
        if preset_name not in PROBLEM_SET_PRESETS:
            raise ValueError(_format_unknown_problem_set(preset_name, sorted(PROBLEM_SET_PRESETS)))
        return tuple(_make_selection(name, args.n_var, args.n_obj) for name in PROBLEM_SET_PRESETS[preset_name])

    if getattr(args, "problem", None):
        return (_make_selection(args.problem, args.n_var, args.n_obj),)

    # Fallback or default
    return (_make_selection("zdt1", args.n_var, args.n_obj),)


def resolve_problem_selections(args: Namespace) -> Iterable[ProblemSelection]:
    """
    Return a list of ProblemSelection objects to run.
    Handles 'all', specific sets, or single problems.
    """
    if getattr(args, "problem_set", None):
        if args.problem_set == "all":
            all_names: list[str] = []
            for key, problems in PROBLEM_SET_PRESETS.items():
                if key == "families":
                    continue
                all_names.extend(problems)
            return tuple(_make_selection(name, args.n_var, args.n_obj) for name in all_names)
        return resolve_problem_selection(args)

    if getattr(args, "problem", None):
        return (_make_selection(args.problem, args.n_var, args.n_obj),)

    return (_make_selection("zdt1", args.n_var, args.n_obj),)
