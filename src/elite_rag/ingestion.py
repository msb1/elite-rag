from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace

from elite_rag.chunking import HierarchicalChunker
from elite_rag.embedding import EmbeddingGemma
from elite_rag.missing_documents import MissingDocumentRecorder
from elite_rag.models import ChildChunk, RawDocument
from elite_rag.parsing.base import ParserRegistry
from elite_rag.vector_store import QdrantVectorStore

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestionReport:
    documents_seen: int = 0
    documents_ingested: int = 0
    documents_skipped_empty: int = 0
    documents_skipped_failed: int = 0
    parents_created: int = 0
    children_upserted: int = 0
    batches_upserted: int = 0
    documents_replaced: int = 0


class IngestionEngine:
    def __init__(
        self,
        registry: ParserRegistry,
        chunker: HierarchicalChunker,
        embedder: EmbeddingGemma,
        store: QdrantVectorStore,
        batch_size: int = 64,
        missing_document_recorder: MissingDocumentRecorder | None = None,
    ) -> None:
        self.registry = registry
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
        self.batch_size = batch_size
        self.missing_document_recorder = missing_document_recorder

    def ingest(
        self,
        documents: Iterable[RawDocument],
        replace_existing: bool = False,
        progress: Callable[[IngestionReport], None] | None = None,
    ) -> IngestionReport:
        documents_seen = 0
        documents_ingested = 0
        documents_skipped_empty = 0
        documents_skipped_failed = 0
        parent_count = 0
        child_count = 0
        batch_count = 0
        replaced_count = 0
        pending: list[ChildChunk] = []

        def report_snapshot() -> IngestionReport:
            return IngestionReport(
                documents_seen=documents_seen,
                documents_ingested=documents_ingested,
                documents_skipped_empty=documents_skipped_empty,
                documents_skipped_failed=documents_skipped_failed,
                parents_created=parent_count,
                children_upserted=child_count,
                batches_upserted=batch_count,
                documents_replaced=replaced_count,
            )

        for document in documents:
            documents_seen += 1
            LOGGER.info(
                "ingestion_document_started document_id=%s source=%s title=%r",
                document.document_id,
                document.source,
                document.title,
            )
            try:
                blocks = self.registry.parse(document)
                if blocks:
                    parents, children = self.chunker.chunk(document, blocks)
            except Exception as exc:
                documents_skipped_failed += 1
                self._record_failed_document(document, exc)
                if progress:
                    progress(report_snapshot())
                continue
            if not blocks:
                documents_skipped_empty += 1
                LOGGER.info("ingestion_document_skipped_empty document_id=%s", document.document_id)
                if progress:
                    progress(report_snapshot())
                continue
            if replace_existing:
                self.store.delete_document(document.document_id)
                replaced_count += 1
            parent_text = {parent.parent_id: parent.text for parent in parents}
            for child in children:
                metadata = {**child.metadata, "_parent_context": parent_text[child.parent_id]}
                pending.append(replace(child, metadata=metadata))
                if len(pending) >= self.batch_size:
                    child_count += self._flush(pending)
                    batch_count += 1
                    pending.clear()
            parent_count += len(parents)
            documents_ingested += 1
            LOGGER.info(
                "ingestion_document_completed document_id=%s parents=%s children=%s",
                document.document_id,
                len(parents),
                len(children),
            )
            if progress:
                progress(report_snapshot())
        if pending:
            child_count += self._flush(pending)
            batch_count += 1
        report = report_snapshot()
        if progress:
            progress(report)
        return report

    def _flush(self, chunks: list[ChildChunk]) -> int:
        LOGGER.info("ingestion_batch_embedding_started chunks=%s", len(chunks))
        vectors = self.embedder.embed_documents([chunk.vector_text for chunk in chunks])
        self.store.upsert(chunks, vectors)
        LOGGER.info("ingestion_batch_upsert_completed chunks=%s", len(chunks))
        return len(chunks)

    def _record_failed_document(self, document: RawDocument, exc: Exception) -> None:
        LOGGER.exception(
            "ingestion_document_skipped_failed document_id=%s source=%s error_type=%s",
            document.document_id,
            document.source,
            type(exc).__name__,
        )
        if self.missing_document_recorder is None:
            return
        try:
            entry = self.missing_document_recorder.record(document, exc)
        except Exception:
            LOGGER.exception(
                "missing_document_record_failed document_id=%s file_recorder=%s",
                document.document_id,
                type(self.missing_document_recorder).__name__,
            )
            return
        LOGGER.warning(
            "missing_document_recorded document_id=%s object_key=%s error_type=%s",
            entry.document_id,
            entry.object_key,
            entry.error_type,
        )
