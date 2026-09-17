from __future__ import annotations

import json
from pathlib import Path

from elite_rag.chunking import HierarchicalChunker, approximate_token_count
from elite_rag.ingestion import IngestionEngine
from elite_rag.missing_documents import MissingDocumentRecorder
from elite_rag.models import ChildChunk, RawDocument
from elite_rag.parsing.base import ParserRegistry


class FakeEmbedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class FakeStore:
    def __init__(self) -> None:
        self.chunks: list[ChildChunk] = []
        self.deleted: list[str] = []

    def upsert(self, chunks: list[ChildChunk], vectors: list[list[float]]) -> None:
        assert len(chunks) == len(vectors)
        self.chunks.extend(chunks)

    def delete_document(self, document_id: str) -> None:
        self.deleted.append(document_id)


def test_ingestion_report_accounts_for_changes_and_empty_documents() -> None:
    store = FakeStore()
    engine = IngestionEngine(
        ParserRegistry(),
        HierarchicalChunker(approximate_token_count, child_max_tokens=80, parent_max_tokens=200),
        FakeEmbedder(),  # type: ignore[arg-type]
        store,  # type: ignore[arg-type]
        batch_size=1,
    )
    documents = [
        RawDocument("doc-1", "jira", "Ticket", {"description": "A useful answer."}),
        RawDocument("doc-2", "jira", "Empty", {"description": ""}),
    ]
    report = engine.ingest(documents, replace_existing=True)

    assert report.documents_seen == 2
    assert report.documents_ingested == 1
    assert report.documents_skipped_empty == 1
    assert report.documents_skipped_failed == 0
    assert report.parents_created == 1
    assert report.children_upserted == 1
    assert report.batches_upserted == 1
    assert report.documents_replaced == 1
    assert store.deleted == ["doc-1"]
    assert store.chunks[0].metadata["_parent_context"].endswith("A useful answer.")


def test_document_specific_chunk_failure_is_recorded_and_next_document_ingests(
    tmp_path: Path,
) -> None:
    missing_docs_file = tmp_path / "missing_docs.json"
    store = FakeStore()
    engine = IngestionEngine(
        ParserRegistry(),
        HierarchicalChunker(len, child_max_tokens=150, parent_max_tokens=1_000),
        FakeEmbedder(),  # type: ignore[arg-type]
        store,  # type: ignore[arg-type]
        batch_size=1,
        missing_document_recorder=MissingDocumentRecorder(str(missing_docs_file)),
    )
    documents = [
            RawDocument(
                "bad-doc",
                "jira",
                "x" * 5_000,
                {"description": "Broken export"},
            {"object_key": "docs/jira/bad-doc.txt"},
        ),
        RawDocument("good-doc", "jira", "Good export", {"description": "Useful text."}),
    ]

    report = engine.ingest(documents)

    assert report.documents_seen == 2
    assert report.documents_ingested == 1
    assert report.documents_skipped_failed == 1
    assert len(store.chunks) == 1
    entries = json.loads(missing_docs_file.read_text(encoding="utf-8"))
    assert entries[0]["document_id"] == "bad-doc"
    assert entries[0]["object_key"] == "docs/jira/bad-doc.txt"
    assert entries[0]["error_type"] == "ValueError"


def test_oversized_non_whitespace_token_is_split_without_skipping_document() -> None:
    store = FakeStore()
    engine = IngestionEngine(
        ParserRegistry(),
        HierarchicalChunker(len, child_max_tokens=150, parent_max_tokens=1_000),
        FakeEmbedder(),  # type: ignore[arg-type]
        store,  # type: ignore[arg-type]
        batch_size=1,
    )
    document = RawDocument("long-token", "jira", "Long token", {"description": "x" * 1_500})

    report = engine.ingest([document])

    assert report.documents_ingested == 1
    assert report.documents_skipped_failed == 0
    assert report.children_upserted > 1
