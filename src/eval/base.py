"""Common interfaces and data models for retrieval and ranking systems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Query:
    """Standardized representation of a user query."""

    query_id: str
    room_image_path: str | None = None
    text_query: str = ""
    scene_type: str = "general"  # e.g., living_room, bedroom, dining_room, home_office
    room_features: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateRug:
    """Standardized representation of a catalog rug."""

    rug_id: str
    title: str = ""
    color: str = ""
    pattern: str = ""
    style: str = ""
    size: str = ""
    sizes: list[str] = field(default_factory=list)
    price: float = 0.0
    image_url: str = ""
    tags: str = ""
    body: str = ""
    searchable_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RankResult:
    """Output item from a ranker."""

    rug_id: str
    score: float
    rank: int
    system_name: str = ""
    explanation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseRanker(ABC):
    """Abstract base class for all retrievers and rankers."""

    name: str = "base_ranker"

    @abstractmethod
    def rank(
        self,
        query: Query,
        candidates: list[CandidateRug],
        top_k: int = 10,
    ) -> list[RankResult]:
        """Rank candidate rugs for a given query."""
        pass

    def batch_rank(
        self,
        queries: list[Query],
        candidates: list[CandidateRug],
        top_k: int = 10,
    ) -> list[list[RankResult]]:
        """Batch ranking default fallback."""
        return [self.rank(q, candidates, top_k=top_k) for q in queries]
