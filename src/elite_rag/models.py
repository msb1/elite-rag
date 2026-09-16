from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class RawDocument:
    document_id: str
    source: str
    title: str
    fields: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TopologicalBlock:
    path: tuple[str, ...]
    text: str


@dataclass(frozen=True, slots=True)
class ParentChunk:
    parent_id: str
    document_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ChildChunk:
    child_id: str
    parent_id: str
    document_id: str
    text: str
    vector_text: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class RetrievedParent:
    parent_id: str
    document_id: str
    text: str
    metadata: dict[str, Any]
    vector_score: float | None = None
    rerank_score: float | None = None
