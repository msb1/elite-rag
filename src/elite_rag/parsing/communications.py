from __future__ import annotations

import re

from elite_rag.models import RawDocument, TopologicalBlock
from elite_rag.parsing.base import FieldParser, TopologicalParser, render_value

EMAIL_BOUNDARY = re.compile(
    r"(?m)(?=^(?:-{2,}\s*Original Message\s*-{2,}|On .+? wrote:|From:\s+\S.+)$)"
)
TIMESTAMP_LINE = re.compile(r"(?m)(?=^\[\d{1,2}:\d{2}(?::\d{2})?\]\s*)")


class EmailParser(TopologicalParser):
    def parse(self, document: RawDocument) -> list[TopologicalBlock]:
        blocks: list[TopologicalBlock] = []
        for field_name, value in document.fields.items():
            text = render_value(value)
            parts = [part.strip() for part in EMAIL_BOUNDARY.split(text) if part.strip()]
            for index, part in enumerate(parts):
                label = "Current message" if index == 0 else f"Quoted reply {index}"
                blocks.append(TopologicalBlock(path=(field_name.title(), label), text=part))
        return blocks


class SlackParser(TopologicalParser):
    def parse(self, document: RawDocument) -> list[TopologicalBlock]:
        if "messages" not in document.fields:
            # EnterpriseRAG-Bench intentionally moves a small number of documents
            # between source trees; retain their declared field topology.
            return FieldParser().parse(document)
        messages = render_value(document.fields.get("messages", ""))
        parts = [part.strip() for part in re.split(r"\n\s*\n", messages) if part.strip()]
        return [
            TopologicalBlock(path=("Thread", f"Message {index}"), text=part)
            for index, part in enumerate(parts, 1)
        ]


class TranscriptParser(TopologicalParser):
    def parse(self, document: RawDocument) -> list[TopologicalBlock]:
        blocks: list[TopologicalBlock] = []
        for field_name, value in document.fields.items():
            text = render_value(value).replace("\\n", "\n")
            parts = [part.strip() for part in TIMESTAMP_LINE.split(text) if part.strip()]
            blocks.extend(
                TopologicalBlock(path=(field_name.title(), f"Turn {index}"), text=part)
                for index, part in enumerate(parts, 1)
            )
        return blocks
