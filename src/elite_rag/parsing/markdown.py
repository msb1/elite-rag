from __future__ import annotations

from markdown_it import MarkdownIt
from markdown_it.token import Token

from elite_rag.models import RawDocument, TopologicalBlock
from elite_rag.parsing.base import TopologicalParser, render_value


class MarkdownParser(TopologicalParser):
    """Split Markdown from its CommonMark token tree, preserving source order."""

    def __init__(self) -> None:
        self._parser = MarkdownIt("commonmark")

    def parse(self, document: RawDocument) -> list[TopologicalBlock]:
        blocks: list[TopologicalBlock] = []
        for field_name, value in document.fields.items():
            blocks.extend(self._parse_field(field_name, render_value(value)))
        return blocks

    def _parse_field(self, field_name: str, text: str) -> list[TopologicalBlock]:
        lines = text.splitlines()
        headings = self._headings(self._parser.parse(text))
        if not headings:
            content = text.strip()
            return (
                [TopologicalBlock((field_name.replace("_", " ").title(),), content)]
                if content
                else []
            )

        blocks: list[TopologicalBlock] = []
        introduction = "\n".join(lines[: headings[0][0]]).strip()
        if introduction:
            blocks.append(TopologicalBlock((field_name.replace("_", " ").title(),), introduction))

        stack: list[tuple[int, str]] = []
        for index, (_, body_start, level, title) in enumerate(headings):
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            body_end = headings[index + 1][0] if index + 1 < len(headings) else len(lines)
            body = "\n".join(lines[body_start:body_end]).strip()
            if body:
                blocks.append(TopologicalBlock(tuple(item_title for _, item_title in stack), body))
        return blocks

    @staticmethod
    def _headings(tokens: list[Token]) -> list[tuple[int, int, int, str]]:
        headings: list[tuple[int, int, int, str]] = []
        for index, token in enumerate(tokens):
            if token.type != "heading_open" or token.map is None:
                continue
            title = tokens[index + 1].content.strip() if index + 1 < len(tokens) else "Untitled"
            headings.append((token.map[0], token.map[1], int(token.tag[1]), title))
        return headings
