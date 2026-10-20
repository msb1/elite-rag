from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Callable
from typing import Any

from elite_rag.documents import Passage, Section, SourceDocument
from elite_rag.parsing import sections_for

SENTENCES = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(])|\n(?=\S)")
FENCE = re.compile(r"^\s*(```|~~~)")
NAMESPACE = uuid.UUID("e14cf362-862a-4dfa-9b8d-21f878cc0ad2")


class PassageChunker:
    def __init__(
        self,
        count_tokens: Callable[[str], int],
        target_tokens: int = 320,
        max_tokens: int = 420,
        overlap_tokens: int = 40,
        policy_signature: str = "v2",
    ) -> None:
        self.count_tokens = count_tokens
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.policy_signature = policy_signature

    def chunk(self, document: SourceDocument) -> list[Passage]:
        metadata = _metadata(document)
        revision = _revision(document, self.policy_signature)
        context = self._document_context(document, metadata)
        result: list[Passage] = []
        # HTML emits individual blocks: join adjacent blocks under the same heading.
        sections = []
        for section in sections_for(document):
            if sections and (sections[-1].field_name, sections[-1].path) == (
                section.field_name,
                section.path,
            ):
                previous = sections[-1]
                sections[-1] = Section(
                    previous.field_name, previous.path, previous.text + "\n\n" + section.text
                )
            else:
                sections.append(section)
        for section_index, section in enumerate(sections):
            section_id = str(
                uuid.uuid5(NAMESPACE, f"{document.document_id}:{revision}:section:{section_index}")
            )
            search_start = 0
            title = self._truncate(document.title, 72)
            heading = self._truncate(" > ".join(section.path), 72)
            prefix = f"{context}Title: {title}\nSection: {heading}\n\n"
            budget = self.max_tokens - self.count_tokens(prefix) - 8
            target = min(self.target_tokens, budget)
            if budget < 64:
                raise ValueError(
                    f"Document title or section path is too long: {document.document_id}"
                )
            for section_ordinal, text in enumerate(
                self._split_section(section.text, target, budget)
            ):
                # Offsets refer to parsed section text, not raw HTML/Markdown bytes.
                pattern = r"\s+".join(re.escape(word) for word in text.split())
                match = re.search(pattern, section.text[search_start:]) if pattern else None
                start = search_start + match.start() if match else None
                end = search_start + match.end() if match else None
                if start is not None:
                    search_start = start + 1
                embedding_text = prefix + text
                if self.count_tokens(embedding_text) > self.max_tokens:
                    raise ValueError(f"Passage exceeds embedding limit: {document.document_id}")
                ordinal = len(result)
                point_id = str(
                    uuid.uuid5(NAMESPACE, f"{document.document_id}:{revision}:{ordinal}")
                )
                result.append(
                    Passage(
                        point_id=point_id,
                        document_id=document.document_id,
                        ordinal=ordinal,
                        field_name=section.field_name,
                        section_path=section.path,
                        text=text,
                        embedding_text=embedding_text,
                        token_count=self.count_tokens(embedding_text),
                        metadata={
                            **metadata,
                            "revision": revision,
                            "section_id": section_id,
                            "section_index": section_index,
                            "section_ordinal": section_ordinal,
                            "section_text": section.text,
                            "section_char_start": start,
                            "section_char_end": end,
                            "offset_reference": "parsed_section",
                        },
                    )
                )
        for passage in result:
            passage.metadata["expected_passage_count"] = len(result)
        return result

    def _truncate(self, text: str, limit: int) -> str:
        if self.count_tokens(text) <= limit:
            return text
        low, high, best = 1, len(text), 0
        while low <= high:
            midpoint = (low + high) // 2
            candidate = text[:midpoint].rstrip() + "…"
            if self.count_tokens(candidate) <= limit:
                best = midpoint
                low = midpoint + 1
            else:
                high = midpoint - 1
        return text[:best].rstrip() + "…"

    def _split_section(self, text: str, target: int, maximum: int) -> list[str]:
        atoms: list[str] = []
        for paragraph in self._paragraphs(text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if self._is_fenced_code(paragraph):
                atoms.extend(self._split_code(paragraph, maximum))
                continue
            if self.count_tokens(paragraph) <= target:
                atoms.append(paragraph)
                continue
            for sentence in SENTENCES.split(paragraph):
                sentence = sentence.strip()
                if not sentence:
                    continue
                if self.count_tokens(sentence) <= maximum:
                    atoms.append(sentence)
                else:
                    atoms.extend(self._split_words(sentence, maximum))
        chunks: list[str] = []
        current: list[str] = []
        for atom in atoms:
            candidate = "\n\n".join([*current, atom])
            if current and self.count_tokens(candidate) > target:
                chunks.append("\n\n".join(current))
                overlap = self._overlap(current)
                current = (
                    overlap if self.count_tokens("\n\n".join([*overlap, atom])) <= maximum else []
                )
            current.append(atom)
        if current:
            chunks.append("\n\n".join(current))
        return chunks

    @staticmethod
    def _paragraphs(text: str) -> list[str]:
        """Keep fenced code blocks together even when they contain blank lines."""
        blocks: list[str] = []
        current: list[str] = []
        fence: str | None = None
        for line in text.splitlines():
            marker = FENCE.match(line)
            if marker:
                if fence is None:
                    if current:
                        blocks.append("\n".join(current))
                        current = []
                    fence = marker.group(1)
                elif marker.group(1) == fence:
                    current.append(line)
                    blocks.append("\n".join(current))
                    current = []
                    fence = None
                    continue
            if not line.strip() and fence is None:
                if current:
                    blocks.append("\n".join(current))
                    current = []
            else:
                current.append(line)
        if current:
            blocks.append("\n".join(current))
        return blocks

    def _overlap(self, atoms: list[str]) -> list[str]:
        if not self.overlap_tokens:
            return []
        result: list[str] = []
        for atom in reversed(atoms):
            candidate = [atom, *result]
            if self.count_tokens("\n\n".join(candidate)) > self.overlap_tokens:
                break
            result = candidate
        # Do not create an orphaned tail fragment from a sentence, table row, or list item.
        return result

    def _document_context(self, document: SourceDocument, normalized: dict[str, Any]) -> str:
        source = document.source.strip().lower()
        lines = [f"Source: {source or 'unknown'}"]
        metadata = {str(key).lower(): value for key, value in document.metadata.items()}
        groups = [
            ("Project", ("project_key", "project_name", "project", "project_id")),
            ("Repository", ("repository_full_name", "repository_name", "repository", "repo")),
            ("Issue", ("issue_key", "ticket_key", "jira_key", "issue_id", "pr_number", "number")),
            ("Space", ("space_key", "space_name", "space")),
        ]
        used: set[str] = set()
        added_labels: set[str] = set()
        for label, aliases in groups:
            value = next(
                (metadata[key] for key in aliases if key in metadata and metadata[key]), None
            )
            if isinstance(value, dict):
                nested = {str(key).lower(): item for key, item in value.items()}
                value = next(
                    (nested[key] for key in ("full_name", "key", "name", "id") if nested.get(key)),
                    None,
                )
            if isinstance(value, (list, tuple, set)):
                value = None
            if value is None:
                continue
            rendered = str(value).strip()
            if not rendered or rendered.lower() in used:
                continue
            used.add(rendered.lower())
            lines.append(f"{label}: {self._truncate(rendered, 32)}")
            added_labels.add(label)
        scope = _object_scope(document)
        scope_label = {"github": "Repository", "confluence": "Space"}.get(source, "Project")
        if scope and scope_label not in added_labels and scope.lower() not in used:
            lines.append(f"{scope_label}: {self._truncate(scope, 32)}")
        for key, label in (
            ("document_type", "Type"),
            ("service", "Service"),
            ("environment", "Environment"),
            ("region", "Region"),
            ("version", "Version"),
        ):
            value = normalized.get(key)
            if value:
                lines.append(f"{label}: {self._truncate(str(value), 16)}")
        return "\n".join(lines) + "\n"


    @staticmethod
    def _is_fenced_code(text: str) -> bool:
        lines = text.splitlines()
        return len(lines) >= 2 and bool(FENCE.match(lines[0])) and bool(FENCE.match(lines[-1]))

    def _split_code(self, text: str, maximum: int) -> list[str]:
        if self.count_tokens(text) <= maximum:
            return [text]
        lines = text.splitlines()
        opening = lines[0]
        closing = lines[-1]
        # Some exporters put an oversized attribute block on the opening fence.
        # Keep the code, but drop fence info when delimiters leave no payload budget.
        minimal_opening = "```" if opening.startswith("```") else "~~~"
        minimal_closing = "```" if closing.startswith("```") else "~~~"
        if self.count_tokens(f"{opening}\n{closing}") >= maximum:
            opening, closing = minimal_opening, minimal_closing
        if self.count_tokens(f"{opening}\n{closing}") >= maximum:
            raise ValueError("Passage budget cannot fit code-fence delimiters")
        code_lines = lines[1:-1]
        chunks: list[str] = []
        current: list[str] = []
        for line in code_lines:
            wrapper = f"{opening}\n{closing}"
            line_budget = maximum - self.count_tokens(wrapper)
            if line_budget < 1:
                raise ValueError("Passage budget leaves no room for code content")
            line_parts = (
                [line]
                if self.count_tokens(line) <= line_budget
                else self._split_words(line, line_budget)
            )
            for line_part in line_parts:
                candidate = "\n".join([opening, *current, line_part, closing])
                if current and self.count_tokens(candidate) > maximum:
                    chunks.append("\n".join([opening, *current, closing]))
                    current = [line_part]
                else:
                    current.append(line_part)
        if current:
            chunks.append("\n".join([opening, *current, closing]))
        return chunks

    def _split_words(self, text: str, maximum: int) -> list[str]:
        result: list[str] = []
        current = ""
        for word in text.split():
            if self.count_tokens(word) > maximum:
                if current:
                    result.append(current)
                    current = ""
                result.extend(self._split_unbroken_word(word, maximum))
                continue
            candidate = f"{current} {word}".strip()
            if current and self.count_tokens(candidate) > maximum:
                result.append(current)
                current = word
            else:
                current = candidate
        if current:
            result.append(current)
        return result

    def _split_unbroken_word(self, word: str, maximum: int) -> list[str]:
        parts: list[str] = []
        remaining = word
        while remaining:
            low, high, best = 1, len(remaining), 0
            while low <= high:
                midpoint = (low + high) // 2
                if self.count_tokens(remaining[:midpoint]) <= maximum:
                    best = midpoint
                    low = midpoint + 1
                else:
                    high = midpoint - 1
            if best == 0:
                raise ValueError("One source character exceeds the passage budget")
            parts.append(remaining[:best])
            remaining = remaining[best:]
        return parts


class HierarchicalPassageChunker(PassageChunker):
    """Keep section boundaries and cap each section-body passage at the configured size."""

    def __init__(self, settings: Any, count_tokens: Callable[[str], int]) -> None:
        body_max = settings.multi_chunk_max_tokens
        overlap = min(settings.chunk_overlap_tokens, max(0, body_max - 64))
        super().__init__(
            count_tokens=count_tokens,
            target_tokens=min(settings.chunk_target_tokens, body_max - overlap),
            max_tokens=settings.bge_embedding_max_length,
            overlap_tokens=overlap,
            policy_signature=(
                f"{settings.pipeline_signature}:multi-hierarchical-v1:"
                f"body-{body_max}:overlap-{overlap}"
            ),
        )
        self.body_max = body_max
        self.body_target = max(64, body_max - overlap)

    def _split_section(self, text: str, target: int, maximum: int) -> list[str]:
        return super()._split_section(text, self.body_target, self.body_max)

    def chunk(self, document: SourceDocument) -> list[Passage]:
        passages = super().chunk(document)
        full_text = "\n\n".join(
            f"{name}:\n{value}" if len(document.fields) > 1 else value
            for name, value in document.fields.items()
            if isinstance(value, str) and value.strip()
        )
        for passage in passages:
            passage.metadata["document_full_text"] = full_text
        return passages


def _revision(document: SourceDocument, policy_signature: str) -> str:
    content = json.dumps(
        {
            "title": document.title,
            "fields": document.fields,
            "metadata": document.metadata,
            "policy_signature": policy_signature,
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _metadata(document: SourceDocument) -> dict[str, Any]:
    raw = document.metadata
    project = next(
        (
            raw[key]
            for key in ("project_id", "project", "space", "team", "repo", "channel")
            if raw.get(key)
        ),
        "global",
    )
    if project == "global":
        project = _object_scope(document) or project
    modified = next(
        (
            raw[key]
            for key in ("last_modified", "last_updated", "updated_at", "date", "recorded_at")
            if raw.get(key)
        ),
        "unknown",
    )
    normalized = {}
    aliases = {
        "document_type": ("document_type", "content_type", "issue_type", "type"),
        "service": ("service", "service_name", "product", "product_name"),
        "environment": ("environment", "deployment", "deployment_type"),
        "region": ("region", "cloud_region"),
        "version": ("version", "release_version", "fix_version"),
        "status": ("status", "state"),
        "effective_date": ("effective_date", "valid_from"),
        "canonical_url": ("canonical_url", "url", "html_url", "web_url"),
        "issue_key": ("issue_key", "ticket_key", "jira_key", "key"),
        "repository": ("repository_full_name", "repository", "repo"),
    }
    lower = {str(key).lower(): value for key, value in raw.items()}
    for name, keys in aliases.items():
        value = next((lower[key] for key in keys if lower.get(key)), None)
        if isinstance(value, dict):
            value = next(
                (value[key] for key in ("full_name", "name", "key") if value.get(key)), None
            )
        if isinstance(value, (str, int, float)):
            normalized[name] = str(value).strip()
    # Extract only explicit labels/identifiers from the source, never generated facts.
    source_text = "\n".join(value for value in document.fields.values() if isinstance(value, str))
    for key in ("service", "environment", "region", "version", "status", "effective_date"):
        if key not in normalized:
            label = key.replace("_", "[ _]")
            match = re.search(rf"(?im)^\s*(?:[-*]\s*)?{label}\s*:\s*([^\n]{{1,80}})", source_text)
            if match:
                normalized[key] = match.group(1).strip()
    if "region" not in normalized:
        regions = sorted(
            set(
                re.findall(
                    r"\b(?:us|eu|ap|sa|ca|me|af)-(?:east|west|south|north|central|southeast|northeast)-\d\b",
                    source_text,
                )
            )
        )
        if len(regions) == 1:
            normalized["region"] = regions[0]
    if "document_type" not in normalized:
        for kind in ("incident", "playbook", "policy", "runbook", "prd", "sop"):
            if re.search(rf"\b{kind}\b", document.title, re.I):
                normalized["document_type"] = kind
                break
    return {
        **normalized,
        "source": document.source.lower(),
        "project_id": str(project).strip().lower(),
        "last_modified": str(modified).strip(),
        "title": document.title,
        "object_key": document.object_key,
        "bucket": document.bucket,
        "etag": document.etag,
        "source_metadata": json.loads(json.dumps(raw, default=str, ensure_ascii=False)),
    }


def _object_scope(document: SourceDocument) -> str | None:
    if not document.object_key:
        return None
    parts = document.object_key.strip("/").split("/")
    source = document.source.lower()
    try:
        source_index = next(index for index, part in enumerate(parts) if part.lower() == source)
    except StopIteration:
        return None
    if source_index + 1 >= len(parts):
        return None
    candidate = parts[source_index + 1]
    if candidate.lower().endswith((".json", ".txt", ".md")):
        return None
    return candidate
