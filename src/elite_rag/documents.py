from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class SourceDocument:
    document_id: str
    source: str
    title: str
    fields: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    bucket: str | None = None
    object_key: str | None = None
    etag: str | None = None


@dataclass(frozen=True, slots=True)
class Section:
    field_name: str
    path: tuple[str, ...]
    text: str


@dataclass(frozen=True, slots=True)
class Passage:
    point_id: str
    document_id: str
    ordinal: int
    field_name: str
    section_path: tuple[str, ...]
    text: str
    embedding_text: str
    token_count: int
    metadata: dict[str, Any]
