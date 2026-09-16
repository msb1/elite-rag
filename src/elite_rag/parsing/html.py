from __future__ import annotations

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag

from elite_rag.models import RawDocument, TopologicalBlock
from elite_rag.parsing.base import TopologicalParser, render_value


class HtmlParser(TopologicalParser):
    """Track text under HTML heading nodes without regex-based tag stripping."""

    def parse(self, document: RawDocument) -> list[TopologicalBlock]:
        blocks: list[TopologicalBlock] = []
        for field_name, value in document.fields.items():
            blocks.extend(self._parse_field(field_name, render_value(value)))
        return blocks

    @staticmethod
    def _parse_field(field_name: str, raw: str) -> list[TopologicalBlock]:
        if "<" not in raw or ">" not in raw:
            return HtmlParser._plain_text(field_name, raw)
        blocks: list[TopologicalBlock] = []
        soup = BeautifulSoup(raw, "html.parser")
        headings: list[tuple[int, str]] = []
        body: list[str] = []

        def flush() -> None:
            content = "\n".join(part for part in body if part).strip()
            if content:
                path = tuple(title for _, title in headings) or (
                    field_name.replace("_", " ").title(),
                )
                blocks.append(TopologicalBlock(path=path, text=content))
            body.clear()

        for node in soup.descendants:
            if isinstance(node, Tag) and node.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                flush()
                level = int(node.name[1])
                while headings and headings[-1][0] >= level:
                    headings.pop()
                headings.append((level, node.get_text(" ", strip=True)))
            elif isinstance(node, NavigableString):
                if node.find_parent(["h1", "h2", "h3", "h4", "h5", "h6", "script", "style"]):
                    continue
                text = str(node).strip()
                if text:
                    body.append(text)
        flush()
        return blocks

    @staticmethod
    def _plain_text(field_name: str, text: str) -> list[TopologicalBlock]:
        return (
            [TopologicalBlock(path=(field_name.replace("_", " ").title(),), text=text)]
            if text
            else []
        )
