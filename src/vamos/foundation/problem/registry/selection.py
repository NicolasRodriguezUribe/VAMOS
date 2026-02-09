from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches

from .specs import ProblemSpec, get_problem_specs
from typing import cast

from ..types import ProblemProtocol


@dataclass(frozen=True)
class ProblemSelection:
    """Concrete problem instance choice with resolved dimensions."""

    spec: ProblemSpec
    n_var: int
    n_obj: int

    def instantiate(self) -> ProblemProtocol:
        """Create a new problem instance with the resolved dimensions."""
        return cast(ProblemProtocol, self.spec.factory(self.n_var, self.n_obj))


_PROBLEM_DOCS = "docs/reference/problems.md"
_TROUBLESHOOTING_DOCS = "docs/guide/troubleshooting.md"


def _suggest_names(name: str, options: list[str]) -> list[str]:
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


def make_problem_selection(key: str, *, n_var: int | None = None, n_obj: int | None = None) -> ProblemSelection:
    """Look up a registered problem by *key* and resolve its dimensions.

    Parameters
    ----------
    key : str
        Registered problem name (e.g. ``"zdt1"``, ``"dtlz2"``).
    n_var : int, optional
        Override the default number of decision variables.
    n_obj : int, optional
        Override the default number of objectives (only for problems
        that allow it).

    Returns
    -------
    ProblemSelection
        A frozen selection with the resolved ``spec``, ``n_var``, and
        ``n_obj`` ready to instantiate.

    Raises
    ------
    KeyError
        If *key* does not match any registered problem.  The error
        message includes the full list of valid names and close-match
        suggestions.
    """
    specs = get_problem_specs()
    try:
        spec = specs[key]
    except KeyError as exc:
        available = sorted(specs.keys())
        raise KeyError(_format_unknown_problem(key, available)) from exc

    actual_n_var, actual_n_obj = spec.resolve_dimensions(n_var=n_var, n_obj=n_obj)
    return ProblemSelection(spec=spec, n_var=actual_n_var, n_obj=actual_n_obj)


__all__ = ["ProblemSelection", "make_problem_selection"]
