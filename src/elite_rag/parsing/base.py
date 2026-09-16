from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

from elite_rag.models import RawDocument, TopologicalBlock


def render_value(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(f"- {render_value(item)}" for item in value)
    if isinstance(value, dict):
        return "\n".join(f"{key}: {render_value(item)}" for key, item in value.items())
    return str(value).strip()


class TopologicalParser(ABC):
    @abstractmethod
    def parse(self, document: RawDocument) -> list[TopologicalBlock]:
        """Return ordered, non-overlapping structural blocks."""


class FieldParser(TopologicalParser):
    """Treat declared document fields as authoritative structural boundaries."""

    def parse(self, document: RawDocument) -> list[TopologicalBlock]:
        blocks: list[TopologicalBlock] = []
        for name, value in document.fields.items():
            text = render_value(value)
            if text:
                blocks.append(TopologicalBlock(path=(name.replace("_", " ").title(),), text=text))
        return blocks


class ParserRegistry:
    def __init__(self, default: TopologicalParser | None = None) -> None:
        self._default = default or FieldParser()
        self._parsers: dict[str, TopologicalParser] = {}

    def register(self, sources: Iterable[str], parser: TopologicalParser) -> None:
        for source in sources:
            self._parsers[source.lower()] = parser

    def parse(self, document: RawDocument) -> list[TopologicalBlock]:
        return self._parsers.get(document.source.lower(), self._default).parse(document)
