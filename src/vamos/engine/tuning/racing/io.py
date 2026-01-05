from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from .param_space import ParamSpace
from .random_search_tuner import TrialResult


def filter_active_config(config: Dict[str, Any], param_space: ParamSpace) -> Dict[str, Any]:
    """
    Return a copy of `config` containing only parameters that are active
    according to `param_space.is_active`.

    Parameters declared in the ParamSpace are kept only when active; keys not
    declared in the ParamSpace (e.g., population_size, offspring_size, repair)
    are always preserved.
    """
    active: Dict[str, Any] = {}
    for name, value in config.items():
        if name in param_space.params:
            if param_space.is_active(name, config):
                active[name] = value
        else:
            active[name] = value
    return active


def history_to_dict(
    history: List[TrialResult],
    param_space: ParamSpace,
    *,
    include_raw: bool = False,
) -> List[Dict[str, Any]]:
    """
    Convert TrialResult list into JSON-serializable dicts, filtering out
    inactive parameters using param_space.
    """
    data: List[Dict[str, Any]] = []
    for trial in history:
        clean_config = filter_active_config(trial.config, param_space)
        record: Dict[str, Any] = {
            "trial_id": trial.trial_id,
            "score": trial.score,
            "config": clean_config,
            "details": trial.details,
        }
        if include_raw:
            record["raw_config"] = trial.config
        data.append(record)
    return data


def save_history_json(
    history: List[TrialResult],
    param_space: ParamSpace,
    path: str | Path,
    *,
    include_raw: bool = False,
) -> None:
    """Persist filtered tuning history to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = history_to_dict(history, param_space, include_raw=include_raw)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def save_history_csv(
    history: List[TrialResult],
    param_space: ParamSpace,
    path: str | Path,
    *,
    include_raw: bool = False,
) -> None:
    """Persist filtered tuning history to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        headers = ["trial_id", "score", "config_json"]
        if include_raw:
            headers.append("raw_config_json")
        writer.writerow(headers)
        for trial in history:
            clean_config = filter_active_config(trial.config, param_space)
            row = [trial.trial_id, trial.score, json.dumps(clean_config)]
            if include_raw:
                row.append(json.dumps(trial.config))
            writer.writerow(row)
def save_checkpoint(
    best_configs: List[Dict[str, Any]],
    elite_archive: List[Dict[str, Any]],
    path: str | Path,
    *,
    metadata: Dict[str, Any] | None = None,
) -> None:
    """
    Save a tuning checkpoint with elite configurations for warm restart.

    Args:
        best_configs: List of best configuration dicts found so far.
        elite_archive: List of elite entry dicts (config + score).
        path: Path to save the checkpoint JSON.
        metadata: Optional additional metadata (e.g., stage_index, num_experiments).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "best_configs": best_configs,
        "elite_archive": elite_archive,
        "metadata": metadata or {},
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(checkpoint, fh, indent=2)


def load_checkpoint(path: str | Path) -> Dict[str, Any]:
    """
    Load a tuning checkpoint for warm restart.

    Returns:
        Dictionary with keys:
            - best_configs: List of configuration dicts.
            - elite_archive: List of elite entry dicts.
            - metadata: Any additional metadata.

    The 'best_configs' can be passed to RacingTuner(initial_configs=...).
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


__all__ = [
    "filter_active_config",
    "history_to_dict",
    "save_history_json",
    "save_history_csv",
    "save_checkpoint",
    "load_checkpoint",
]
