from __future__ import annotations

import json
import re
from typing import Any

from bs4 import BeautifulSoup, NavigableString, Tag
from markdown_it import MarkdownIt

from elite_rag.documents import Section, SourceDocument

HEADING = re.compile(r"^h([1-6])$")
TABLE_SEPARATOR = re.compile(r"^:?-{3,}:?$")
BLOCK_TAGS = {"blockquote", "div", "dl", "figure", "ol", "p", "pre", "table", "ul"}
SOURCE_LABELS = (
    "Issue summary",
    "Summary of changes",
    "Implementation details",
    "Implementation notes",
    "Expected behavior",
    "Actual behavior",
    "Observed behavior",
    "Acceptance criteria",
    "Compatibility notes",
    "Backwards compatibility",
    "Backward compatibility",
    "Steps to reproduce",
    "Reproduction steps",
    "Root cause",
    "Contributing factors",
    "Action items",
    "Rollout plan",
    "Rollback plan",
    "Files changed",
    "Test plan",
    "Verification",
    "Investigation",
    "Mitigation",
    "Workaround",
    "Resolution",
    "Impact",
    "Timeline",
    "Environment",
    "Motivation",
    "Implementation",
    "Validation",
    "Testing",
    "Tests",
    "Summary",
    "Context",
    "Decision",
    "Risks",
    "Follow-up",
    "Follow up",
)
SOURCE_LABELS_BY_SOURCE = {
    "jira": {
        "Issue summary",
        "Steps to reproduce",
        "Reproduction steps",
        "Environment",
        "Expected behavior",
        "Actual behavior",
        "Observed behavior",
        "Root cause",
        "Investigation",
        "Resolution",
        "Mitigation",
        "Workaround",
        "Impact",
        "Timeline",
        "Acceptance criteria",
        "Action items",
        "Follow-up",
        "Follow up",
    },
    "github": {
        "Motivation",
        "Summary of changes",
        "Implementation",
        "Implementation details",
        "Implementation notes",
        "Backwards compatibility",
        "Backward compatibility",
        "Compatibility notes",
        "Files changed",
        "Tests",
        "Testing",
        "Test plan",
        "Validation",
        "Rollout plan",
        "Rollback plan",
        "Risks",
        "Context",
        "Decision",
        "Follow-up",
        "Follow up",
    },
}


def render_value(value: Any) -> str:
    """Render structured source values without losing field or list boundaries."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        rendered = [render_value(item) for item in value]
        return "\n".join(f"- {item}" for item in rendered if item)
    if isinstance(value, dict):
        rendered = []
        for key, item in value.items():
            text = render_value(item)
            if text:
                rendered.append(f"{_label(str(key))}: {text}")
        return "\n".join(rendered)
    return str(value)


def sections_for(document: SourceDocument) -> list[Section]:
    """Extract heading-aware sections while retaining structured source records."""
    sections: list[Section] = []
    for field_name, value in document.fields.items():
        field_label = _label(field_name)
        if isinstance(value, list) and value and any(isinstance(item, dict) for item in value):
            for index, item in enumerate(value, start=1):
                item_label = _record_label(item, index)
                path = (field_label, item_label)
                sections.extend(
                    _sections_from_text(field_name, render_value(item), path, document.source)
                )
            continue

        sections.extend(
            _sections_from_text(field_name, render_value(value), (field_label,), document.source)
        )
    return sections


def _sections_from_text(
    field_name: str, text: str, base_path: tuple[str, ...], source: str
) -> list[Section]:
    text = text.strip()
    if not text:
        return []
    if _looks_like_html(text):
        return _html_sections(field_name, text, base_path)
    markdown_sections = _markdown_sections(
        field_name,
        _normalize_markdown_tables(text),
        base_path,
    )
    result: list[Section] = []
    for section in markdown_sections:
        result.extend(_source_labeled_sections(field_name, section.text, section.path, source))
    return result


def _source_labeled_sections(
    field_name: str, text: str, base_path: tuple[str, ...], source: str
) -> list[Section]:
    """Split common ticket/PR field labels into independently searchable sections."""
    labels = SOURCE_LABELS_BY_SOURCE.get(source.lower(), set(SOURCE_LABELS))
    pattern = re.compile(
        r"(?<![A-Za-z0-9])(?P<label>"
        + "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
        + r")\s*:",
        flags=re.IGNORECASE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return [Section(field_name, base_path, text)]

    result: list[Section] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        result.append(Section(field_name, base_path, preamble))
    display_labels = {label.casefold(): label for label in SOURCE_LABELS}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip(" \t\r\n:;.-")
        if body:
            label = display_labels[match.group("label").casefold()]
            result.append(Section(field_name, (*base_path, label), body))
    return result or [Section(field_name, base_path, text)]


def _looks_like_html(text: str) -> bool:
    lower = text[:2000].lower()
    return any(tag in lower for tag in ("<p", "<div", "<h1", "<h2", "<ul", "<table"))


def _html_sections(field_name: str, text: str, base_path: tuple[str, ...]) -> list[Section]:
    soup = BeautifulSoup(text, "html.parser")
    for node in soup(["script", "style", "nav", "noscript"]):
        node.decompose()

    headings: list[tuple[int, str]] = []
    sections: list[Section] = []

    def current_path() -> tuple[str, ...]:
        return (*base_path, *(title for _, title in headings))

    def add(text_value: str) -> None:
        body = text_value.strip()
        if body:
            sections.append(Section(field_name, current_path(), body))

    def walk(node: Tag) -> None:
        for child in node.children:
            if isinstance(child, NavigableString):
                body = str(child).strip()
                if body:
                    add(body)
                continue
            if not isinstance(child, Tag):
                continue

            name = child.name.lower()
            heading = HEADING.fullmatch(name)
            if heading:
                level = int(heading.group(1))
                title = child.get_text(" ", strip=True)
                while headings and headings[-1][0] >= level:
                    headings.pop()
                if title:
                    headings.append((level, title))
            elif name == "table":
                caption_node = child.find("caption")
                caption = caption_node.get_text(" ", strip=True) if caption_node else ""
                table_path = current_path()
                if caption:
                    table_path = (*table_path, f"Table: {caption}")
                table_text = _html_table_text(child)
                if table_text:
                    sections.append(Section(field_name, table_path, table_text))
            elif name in {"ul", "ol"}:
                add(_html_list_text(child))
            elif name == "pre":
                code = child.get_text("\n", strip=False).strip()
                language = child.get("data-language") or child.get("class")
                label = " ".join(language) if isinstance(language, list) else language
                fence = f"```{label or ''}\n{code}\n```"
                add(fence)
            elif name == "p":
                add(child.get_text(" ", strip=True))
            elif name in {"script", "style", "nav", "noscript"}:
                continue
            elif any(
                isinstance(descendant, Tag) and descendant.name in BLOCK_TAGS
                for descendant in child.children
            ):
                walk(child)
            else:
                add(child.get_text(" ", strip=True))

    walk(soup)
    return sections


def _html_table_text(table: Tag) -> str:
    rows = table.find_all("tr")
    if not rows:
        return ""
    header_index: int | None = None
    for index, row in enumerate(rows):
        if row.find("th", recursive=False):
            header_index = index
            break

    headers: list[str] = []
    if header_index is not None:
        headers = [
            cell.get_text(" ", strip=True)
            for cell in rows[header_index].find_all(["th", "td"], recursive=False)
        ]

    output: list[str] = []
    for row_index, row in enumerate(rows):
        if row_index == header_index:
            continue
        cells = [
            cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"], recursive=False)
        ]
        if not any(cells):
            continue
        if headers:
            labels = [
                (headers[index] if index < len(headers) else "") or f"Column {index + 1}"
                for index in range(len(cells))
            ]
            pairs = [f"{labels[index]}: {value}" for index, value in enumerate(cells)]
        else:
            pairs = [f"Column {index + 1}: {value}" for index, value in enumerate(cells)]
        output.append(" | ".join(pairs))
    return "\n".join(output)


def _html_list_text(list_node: Tag) -> str:
    rows: list[str] = []
    for index, item in enumerate(list_node.find_all("li", recursive=False), start=1):
        text = item.get_text(" ", strip=True)
        if text:
            rows.append(f"{str(index) + '.' if list_node.name == 'ol' else '-'} {text}")
    return "\n".join(rows)


def _normalize_markdown_tables(text: str) -> str:
    """Repeat Markdown table headers as labels so each retrieved row is self-contained."""
    lines = text.splitlines()
    output: list[str] = []
    in_fence = False
    index = 0
    while index < len(lines):
        line = lines[index]
        if re.match(r"^\s*(```|~~~)", line):
            in_fence = not in_fence
            output.append(line)
            index += 1
            continue
        if (
            not in_fence
            and index + 1 < len(lines)
            and "|" in line
            and _is_markdown_separator(lines[index + 1])
        ):
            headers = _pipe_cells(line)
            index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                values = _pipe_cells(lines[index])
                pairs = []
                for cell_index, value in enumerate(values):
                    if not value:
                        continue
                    label = (
                        headers[cell_index]
                        if cell_index < len(headers)
                        else f"Column {cell_index + 1}"
                    )
                    pairs.append(f"{label}: {value}")
                if pairs:
                    output.append(" | ".join(pairs))
                index += 1
            continue
        output.append(line)
        index += 1
    return "\n".join(output)


def _pipe_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_markdown_separator(line: str) -> bool:
    cells = _pipe_cells(line)
    return bool(cells) and all(TABLE_SEPARATOR.fullmatch(cell) for cell in cells)


def _markdown_sections(
    field_name: str, text: str, base_path: tuple[str, ...] | None = None
) -> list[Section]:
    lines = text.splitlines()
    tokens = MarkdownIt("commonmark").parse(text)
    headings: list[tuple[int, int, int, str]] = []
    for index, token in enumerate(tokens):
        if token.type == "heading_open" and token.map:
            title = tokens[index + 1].content.strip() if index + 1 < len(tokens) else ""
            headings.append((token.map[0], token.map[1], int(token.tag[1]), title))
    root = base_path or (_label(field_name),)
    if not headings:
        body = text.strip()
        return [Section(field_name, root, body)] if body else []

    result: list[Section] = []
    preamble = "\n".join(lines[: headings[0][0]]).strip()
    if preamble:
        result.append(Section(field_name, root, preamble))
    stack: list[tuple[int, str]] = []
    for index, (_, body_start, level, title) in enumerate(headings):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        body_end = headings[index + 1][0] if index + 1 < len(headings) else len(lines)
        body = "\n".join(lines[body_start:body_end]).strip()
        if body:
            result.append(Section(field_name, (*root, *(name for _, name in stack)), body))
    return result


def _record_label(value: dict[str, Any], index: int) -> str:
    candidates = ("title", "summary", "subject", "name", "issue_key", "key", "author", "created_at")
    labels = []
    normalized = {str(key).lower(): item for key, item in value.items()}
    for key in candidates:
        item = normalized.get(key)
        if item is None or isinstance(item, (dict, list)):
            continue
        rendered = str(item).strip()
        if rendered and rendered not in labels:
            labels.append(rendered[:48])
        if len(labels) == 2:
            break
    suffix = " · ".join(labels)
    return f"Entry {index}" + (f": {suffix}" if suffix else "")


def _label(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[_-]+", " ", value)).strip().title()


def source_from_json(data: dict[str, Any], source: str, object_key: str) -> SourceDocument:
    title_field = data.get("title_field_name")
    content_fields = data.get("content_field_names")
    if not isinstance(title_field, str) or title_field not in data:
        raise ValueError(f"{object_key}: invalid title_field_name")
    if not isinstance(content_fields, list) or not content_fields:
        raise ValueError(f"{object_key}: invalid content_field_names")
    if any(not isinstance(name, str) or name not in data for name in content_fields):
        raise ValueError(f"{object_key}: missing declared content field")
    excluded = {*content_fields, "title_field_name", "content_field_names"}
    return SourceDocument(
        document_id=str(data.get("dataset_doc_uuid") or object_key.rsplit("/", 1)[-1]),
        source=source.lower(),
        title=str(data[title_field]),
        fields={name: data[name] for name in content_fields},
        metadata={key: value for key, value in data.items() if key not in excluded},
        object_key=object_key,
    )


def source_from_bytes(body: bytes, source: str, object_key: str) -> SourceDocument:
    if object_key.lower().endswith(".json"):
        data = json.loads(body)
        if not isinstance(data, dict):
            raise ValueError(f"{object_key}: expected a JSON object")
        return source_from_json(data, source, object_key)
    text = body.decode("utf-8")
    lines = text.splitlines()
    title_index = next((index for index, line in enumerate(lines) if line.strip()), None)
    title = lines[title_index].strip().lstrip("# ") if title_index is not None else ""
    content = (
        "\n".join([*lines[:title_index], *lines[title_index + 1 :]]).strip()
        if title_index is not None
        else text
    )
    name = object_key.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return SourceDocument(
        document_id=name.split("__", 1)[0] if name.startswith("dsid_") else name,
        source=source.lower(),
        title=title or name,
        fields={"content": content},
        object_key=object_key,
    )
