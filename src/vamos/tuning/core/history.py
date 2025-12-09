from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

from .validation import ConfigSpec


JsonLike = Union[Dict[str, Any], List[Any], int, float, str, bool, None]


@dataclass
class TrialRecord:
    """
    Generic trial record loaded from tuning history.
    """

    trial_id: int
    config: Mapping[str, Any]
    score: float
    details: Mapping[str, Any] = field(default_factory=dict)
    source: str | None = None


def _normalize_trial_object(obj: Mapping[str, Any], *, source: str | None = None) -> TrialRecord:
    """
    Normalize a raw trial dict into a TrialRecord, validating required fields.
    """
    if "trial_id" not in obj or "config" not in obj or "score" not in obj:
        raise ValueError("Trial object must contain 'trial_id', 'config', and 'score' fields")

    trial_id = int(obj["trial_id"])
    config = obj["config"]
    if not isinstance(config, Mapping):
        raise ValueError("Trial 'config' must be a mapping")

    score = float(obj["score"])
    details = obj.get("details", {})
    if details is None:
        details = {}
    if not isinstance(details, Mapping):
        details = {"details": details}

    return TrialRecord(
        trial_id=trial_id,
        config=dict(config),
        score=score,
        details=dict(details),
        source=source,
    )


def load_history_json(path: Union[str, Path]) -> List[TrialRecord]:
    """
    Load a history JSON file (list of trial dicts) into TrialRecord objects.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise ValueError("History JSON must be a list of trial objects")

    records: List[TrialRecord] = []
    for obj in data:
        if not isinstance(obj, Mapping):
            raise ValueError("Each trial entry must be a mapping/dict")
        record = _normalize_trial_object(obj, source=str(p))
        records.append(record)
    return records


def load_histories_from_directory(path: Union[str, Path]) -> List[TrialRecord]:
    """
    Load all trial records from a directory of JSON files (or a single file).
    """
    p = Path(path)
    if p.is_file():
        return load_history_json(p)

    if not p.is_dir():
        raise ValueError(f"Path does not exist or is not a file/directory: {p}")

    all_records: List[TrialRecord] = []
    for file_path in sorted(p.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() == ".json":
            records = load_history_json(file_path)
            all_records.extend(records)

    return all_records


def _config_key(config: Mapping[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    """
    Build a deterministic key representing a configuration for uniqueness.
    """
    items: List[Tuple[str, Any]] = []
    for k, v in config.items():
        try:
            hash(v)
            value = v
        except Exception:
            value = repr(v)
        items.append((str(k), value))
    items.sort(key=lambda kv: kv[0])
    return tuple(items)


def select_top_k_trials(
    records: Sequence[TrialRecord],
    k: int,
    maximize: bool = True,
    unique_by_config: bool = True,
) -> List[TrialRecord]:
    """
    Select the top-k trials from a list of TrialRecord objects.
    """
    if k <= 0 or not records:
        return []

    sorted_records = sorted(records, key=lambda r: r.score, reverse=maximize)

    if not unique_by_config:
        return sorted_records[:k]

    selected: List[TrialRecord] = []
    seen_keys: set[Tuple[Tuple[str, Any], ...]] = set()

    for r in sorted_records:
        key = _config_key(r.config)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        selected.append(r)
        if len(selected) >= k:
            break

    return selected


def make_config_specs_from_trials(
    trials: Sequence[TrialRecord],
    label_prefix: str,
    include_score: bool = True,
    include_trial_id: bool = True,
    score_format: str = ".4f",
) -> List[ConfigSpec]:
    """
    Convert trial records into ConfigSpec objects with human-friendly labels.
    """
    specs: List[ConfigSpec] = []

    for r in trials:
        parts: List[str] = [label_prefix]

        if include_trial_id:
            parts.append(f"id={r.trial_id}")

        if include_score:
            try:
                formatted_score = format(r.score, score_format)
            except Exception:
                formatted_score = str(r.score)
            parts.append(f"score={formatted_score}")

        label = " ".join(parts)
        specs.append(ConfigSpec(label=label, config=dict(r.config)))

    return specs


def load_top_k_as_config_specs(
    path: Union[str, Path],
    k: int,
    maximize: bool = True,
    label_prefix: Optional[str] = None,
    unique_by_config: bool = True,
) -> List[ConfigSpec]:
    """
    Load a history JSON, select top-k trials, and convert them to ConfigSpec objects.
    """
    p = Path(path)
    records = load_history_json(p)
    top_trials = select_top_k_trials(records, k=k, maximize=maximize, unique_by_config=unique_by_config)

    if label_prefix is None:
        label_prefix = f"history:{p.name}"

    return make_config_specs_from_trials(top_trials, label_prefix=label_prefix)


def example_history_usage() -> None:
    """
    Example of how to use the history utilities to create baselines for validation.
    This is a usage example only and should not be executed on import.
    """
    # path = "results/autonsga_zdt1_phase2_history.json"
    # records = load_history_json(path)
    # top_trials = select_top_k_trials(records, k=3, maximize=True, unique_by_config=True)
    # specs = make_config_specs_from_trials(
    #     top_trials,
    #     label_prefix="AutoNSGA-II (racing)",
    #     include_score=True,
    #     include_trial_id=True,
    # )
    # for spec in specs:
    #     print(spec.label)
    #     print("  config:", spec.config)


__all__ = [
    "TrialRecord",
    "JsonLike",
    "load_history_json",
    "load_histories_from_directory",
    "select_top_k_trials",
    "make_config_specs_from_trials",
    "load_top_k_as_config_specs",
    "example_history_usage",
]
