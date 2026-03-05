"""
Persistent knowledge base for MOEA tuning results.

Stores tuning results keyed by problem features so that future tuning runs
on similar problems can warm-start from proven configurations instead of
starting from scratch.

Storage format is a JSON file at ``~/.vamos/tuning_knowledge.json``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .problem_features import ProblemFeatures


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


DEFAULT_KB_PATH = Path.home() / ".vamos" / "tuning_knowledge.json"


@dataclass
class KnowledgeEntry:
    """A single tuning result stored in the knowledge base."""

    problem_features: ProblemFeatures
    algorithm: str
    encoding: str
    best_config: dict[str, Any]
    score: float
    tuning_metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_features": self.problem_features.to_dict(),
            "algorithm": self.algorithm,
            "encoding": self.encoding,
            "best_config": self.best_config,
            "score": self.score,
            "tuning_metadata": self.tuning_metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> KnowledgeEntry:
        return cls(
            problem_features=ProblemFeatures.from_dict(d["problem_features"]),
            algorithm=d["algorithm"],
            encoding=d["encoding"],
            best_config=d["best_config"],
            score=d["score"],
            tuning_metadata=d.get("tuning_metadata", {}),
            timestamp=d.get("timestamp", ""),
        )


class KnowledgeBase:
    """Persistent store for MOEA tuning knowledge.

    Parameters
    ----------
    path : Path, optional
        Path to the JSON knowledge base file.
        Defaults to ``~/.vamos/tuning_knowledge.json``.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_KB_PATH
        self._entries: list[KnowledgeEntry] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._entries = []
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._entries = [KnowledgeEntry.from_dict(e) for e in data]
            _logger().debug("Loaded %d entries from %s", len(self._entries), self.path)
        except Exception:
            _logger().warning("Failed to load knowledge base from %s", self.path, exc_info=True)
            self._entries = []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [e.to_dict() for e in self._entries]
        self.path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def store(
        self,
        features: ProblemFeatures,
        algorithm: str,
        encoding: str,
        best_config: dict[str, Any],
        score: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a tuning result."""
        entry = KnowledgeEntry(
            problem_features=features,
            algorithm=algorithm,
            encoding=encoding,
            best_config=best_config,
            score=score,
            tuning_metadata=metadata or {},
        )
        self._entries.append(entry)
        self._save()
        _logger().info(
            "Stored tuning result: algorithm=%s, score=%.4f, features=%s",
            algorithm,
            score,
            features,
        )

    def query_similar(
        self,
        features: ProblemFeatures,
        algorithm: str | None = None,
        encoding: str | None = None,
        k: int = 5,
    ) -> list[KnowledgeEntry]:
        """Find the K most similar entries by feature distance.

        Parameters
        ----------
        features : ProblemFeatures
            Target problem features.
        algorithm : str, optional
            Filter by algorithm name (exact match).
        encoding : str, optional
            Filter by encoding (exact match).
        k : int
            Number of results to return.

        Returns
        -------
        list of KnowledgeEntry
            Sorted by distance (closest first).
        """
        candidates = self._entries
        if algorithm is not None:
            candidates = [e for e in candidates if e.algorithm == algorithm]
        if encoding is not None:
            candidates = [e for e in candidates if e.encoding == encoding]

        scored = [(e, features.distance(e.problem_features)) for e in candidates]
        scored.sort(key=lambda x: x[1])
        return [e for e, _ in scored[:k]]

    def get_warm_start_configs(
        self,
        features: ProblemFeatures,
        algorithm: str | None = None,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """Get warm-start configurations for a new tuning run.

        Returns the best_config from the K nearest entries.
        """
        entries = self.query_similar(features, algorithm=algorithm, k=k)
        return [e.best_config for e in entries]

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> list[KnowledgeEntry]:
        return list(self._entries)


__all__ = ["KnowledgeBase", "KnowledgeEntry", "DEFAULT_KB_PATH"]
