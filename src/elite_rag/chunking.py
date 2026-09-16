from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Iterable
from typing import Any

from elite_rag.models import ChildChunk, ParentChunk, RawDocument, TopologicalBlock

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[])|\n+(?=\S)")
UUID_NAMESPACE = uuid.UUID("6db17d97-72e9-4bad-9817-4c3d279453ce")


class HierarchicalChunker:
    """Create bounded parent/child chunks while never reusing source text."""

    def __init__(
        self,
        count_tokens: Callable[[str], int],
        child_max_tokens: int = 150,
        parent_max_tokens: int = 2000,
    ) -> None:
        if child_max_tokens >= parent_max_tokens:
            raise ValueError("child_max_tokens must be smaller than parent_max_tokens")
        self.count_tokens = count_tokens
        self.child_max_tokens = child_max_tokens
        self.parent_max_tokens = parent_max_tokens

    def chunk(
        self, document: RawDocument, blocks: Iterable[TopologicalBlock]
    ) -> tuple[list[ParentChunk], list[ChildChunk]]:
        metadata = normalize_metadata(document)
        prefix = self._parent_prefix(document, metadata)
        parent_budget = self.parent_max_tokens - self.count_tokens(prefix)
        if parent_budget < self.child_max_tokens:
            raise ValueError("Parent metadata leaves insufficient room for child content")

        units: list[TopologicalBlock] = []
        for block in blocks:
            structural_overhead = self.count_tokens(
                self._render_block(TopologicalBlock(block.path, ""))
            )
            unit_budget = parent_budget - structural_overhead
            if unit_budget < self.child_max_tokens:
                raise ValueError("Parent section labels leave insufficient room for child content")
            for piece in self._split_naturally(block.text, unit_budget):
                units.append(TopologicalBlock(path=block.path, text=piece))

        parent_groups = self._pack_parent_units(units, parent_budget)
        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []
        child_number = 0
        for parent_number, group in enumerate(parent_groups):
            parent_id = stable_id(document.document_id, "parent", parent_number)
            parent_body = "\n\n".join(self._render_block(item) for item in group)
            parent_text = f"{prefix}\n\n{parent_body}".strip()
            parents.append(
                ParentChunk(
                    parent_id=parent_id,
                    document_id=document.document_id,
                    text=parent_text,
                    metadata=metadata,
                )
            )

            for block in group:
                child_prefix = self._child_prefix(document, metadata, block.path)
                child_budget = self.child_max_tokens - self.count_tokens(child_prefix)
                if child_budget < 8:
                    raise ValueError("Child metadata leaves insufficient room for content")
                for piece in self._split_naturally(block.text, child_budget):
                    child_text = self._render_block(TopologicalBlock(block.path, piece))
                    vector_text = f"{child_prefix}{piece}".strip()
                    children.append(
                        ChildChunk(
                            child_id=stable_id(document.document_id, "child", child_number),
                            parent_id=parent_id,
                            document_id=document.document_id,
                            text=child_text,
                            vector_text=vector_text,
                            metadata=metadata,
                        )
                    )
                    child_number += 1
        return parents, children

    def _pack_parent_units(
        self, units: list[TopologicalBlock], budget: int
    ) -> list[list[TopologicalBlock]]:
        groups: list[list[TopologicalBlock]] = []
        current: list[TopologicalBlock] = []
        for unit in units:
            candidate = [*current, unit]
            rendered = "\n\n".join(self._render_block(item) for item in candidate)
            if current and self.count_tokens(rendered) > budget:
                groups.append(current)
                current = [unit]
            else:
                current = candidate
        if current:
            groups.append(current)
        return groups

    def _split_naturally(self, text: str, budget: int) -> list[str]:
        if not text.strip():
            return []
        if self.count_tokens(text) <= budget:
            return [text.strip()]

        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        atoms: list[str] = []
        for paragraph in paragraphs:
            if self.count_tokens(paragraph) <= budget:
                atoms.append(paragraph)
                continue
            sentences = [
                part.strip() for part in SENTENCE_BOUNDARY.split(paragraph) if part.strip()
            ]
            for sentence in sentences:
                if self.count_tokens(sentence) <= budget:
                    atoms.append(sentence)
                else:
                    atoms.extend(self._split_oversized_atom(sentence, budget))
        return self._pack_atoms(atoms, budget)

    def _split_oversized_atom(self, text: str, budget: int) -> list[str]:
        words = text.split()
        chunks: list[str] = []
        current: list[str] = []
        for word in words:
            if current and self.count_tokens(" ".join([*current, word])) > budget:
                chunks.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            chunks.append(" ".join(current))
        if any(self.count_tokens(chunk) > budget for chunk in chunks):
            raise ValueError("A single token-like unit exceeds the configured chunk budget")
        return chunks

    def _pack_atoms(self, atoms: list[str], budget: int) -> list[str]:
        chunks: list[str] = []
        current = ""
        for atom in atoms:
            candidate = f"{current}\n\n{atom}".strip()
            if current and self.count_tokens(candidate) > budget:
                chunks.append(current)
                current = atom
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _render_block(block: TopologicalBlock) -> str:
        return f"SECTION: {' > '.join(block.path)}\n{block.text}" if block.path else block.text

    @staticmethod
    def _parent_prefix(document: RawDocument, metadata: dict[str, Any]) -> str:
        return (
            f"DOCUMENT ID: {document.document_id}\nTITLE: {document.title}\n"
            f"SOURCE: {metadata['source']}\nPROJECT: {metadata['project_id']}\n"
            f"LAST MODIFIED: {metadata['last_modified']}"
        )

    @staticmethod
    def _child_prefix(
        document: RawDocument, metadata: dict[str, Any], path: tuple[str, ...]
    ) -> str:
        return (
            f"title: {document.title} | text: "
            f"source: {metadata['source']} | project: {metadata['project_id']} | "
            f"updated: {metadata['last_modified'][:7]} | "
            f"section: {' > '.join(path)} | content: "
        )


def normalize_metadata(document: RawDocument) -> dict[str, Any]:
    raw = document.metadata
    project = next(
        (
            raw[key]
            for key in ("project_id", "project", "space", "team", "repo", "channel", "drive_area")
            if raw.get(key) not in (None, "")
        ),
        "global",
    )
    modified = next(
        (
            raw[key]
            for key in (
                "last_modified",
                "last_updated",
                "updated_at",
                "date",
                "last_email_at",
                "recorded_at",
            )
            if raw.get(key) not in (None, "")
        ),
        "unknown",
    )
    result = {key: _json_metadata(value) for key, value in raw.items()}
    result.update(
        {
            "source": document.source.lower(),
            "project_id": _metadata_value(project).lower(),
            "last_modified": _metadata_value(modified),
            "title": document.title,
        }
    )
    return result


def _metadata_value(value: object) -> str:
    return str(value).strip()


def _json_metadata(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_metadata(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_metadata(item) for key, item in value.items()}
    return str(value)


def stable_id(document_id: str, level: str, index: int) -> str:
    return str(uuid.uuid5(UUID_NAMESPACE, f"{document_id}:{level}:{index}"))


def approximate_token_count(text: str) -> int:
    """Dependency-free counter for dry runs and tests, not production sizing."""
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))
